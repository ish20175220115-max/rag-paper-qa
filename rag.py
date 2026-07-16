import dashscope
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from knowledge import get_vector_store, get_parent_store, get_grandparent_store
from config import DASHSCOPE_API_KEY
import time
import os


# ── 用 RunnableLambda 调用 DashScope（避免 BaseChatModel 兼容问题）──
def _call_qwen(messages: list) -> AIMessage:
    """将 LangChain 消息列表发给百炼 qwen-max，返回 AIMessage"""
    payload = []
    for m in messages:
        t = getattr(m, "type", "")
        if t == "system":
            payload.append({"role": "system", "content": m.content})
        elif t == "ai":
            payload.append({"role": "assistant", "content": m.content})
        elif t == "human":
            payload.append({"role": "user", "content": m.content})
        else:
            # fallback: treat as user message
            payload.append({"role": "user", "content": str(m.content)})

    resp = dashscope.Generation.call(
        api_key=DASHSCOPE_API_KEY,
        model="qwen-max",
        messages=payload,
        result_format="message",
    )
    if resp.status_code == 200 and resp.output and resp.output.choices:
        return AIMessage(content=resp.output.choices[0].message.content)
    else:
        return AIMessage(content=f"调用失败: {resp.code} {resp.message}")


def _format_docs(docs: list, max_chars: int = 100000) -> str:
    """
    将文档列表格式化为上下文字符串，按字符数动态截断。
    优先保留排序靠前的父块（按 chunk_index 已排序），保证完整章节优先。
    """
    if not docs:
        return "（无相关信息）"

    parts = []
    total_len = 0

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知")
        header = (
            doc.metadata.get("H1", "")
            or doc.metadata.get("H2", "")
            or doc.metadata.get("H3", "")
        )
        header_str = f"[章节: {header}] " if header else ""
        part = f"[{i+1}] {header_str}来源: {source}\n{doc.page_content}"

        # 检查加上当前块是否会超限
        if total_len + len(part) > max_chars:
            remaining = len(docs) - i
            parts.append(f"[{i+1}] ...(剩余 {remaining} 个片段因长度限制未显示)")
            break

        parts.append(part)
        total_len += len(part)

    return "\n\n---\n\n".join(parts)


def _retrieve(question: str):
    """
    返回 (docs, min_score)
    docs: 检索到的父块列表（优先章级父块，若无则降级到节级父块）
    min_score: 加权后最小的 L2 距离（来自子块，用于阈值判断）
    """
    # 调试：打印进程ID和问题，区分热重载
    print(f"[检索] 进程ID: {os.getpid()} | 时间戳: {time.time():.3f} | 问题: {question[:40]}...")

    try:
        child_vs = get_vector_store()
        parent_store = get_parent_store()
        grandparent_store = get_grandparent_store()

        # 意图识别：是否在问参考文献（用于调整召回数）
        ref_keywords = ["参考", "引用", "文献", "参考文献", "引用了", "几篇", "多少篇"]
        is_ref_query = any(kw in question for kw in ref_keywords)
        k = 30 if is_ref_query else 15

        # 第一步：子块向量检索
        docs_with_scores = child_vs.similarity_search_with_score(question, k=k)
        if not docs_with_scores:
            return [], 999.0

        print(f"\n[检索] {question[:60]} | 参考文献意图: {is_ref_query}")

        # 调试：打印前3个子块的 grandparent_id
        print("[调试] 命中前3个子块的元数据：")
        for idx, (doc, score) in enumerate(docs_with_scores[:3]):
            gp_id = doc.metadata.get("grandparent_id")
            p_id = doc.metadata.get("parent_id")
            print(f"  子块{idx+1}: grandparent_id={gp_id}, parent_id={p_id}")

        # 位置权重计算（仅影响排序和 min_score）
        weighted = []
        for doc, score in docs_with_scores:
            chunk_idx = doc.metadata.get("chunk_index", 0)
            total = doc.metadata.get("total_chunks", 1)
            is_ref = doc.metadata.get("is_reference", False)
            weight = 1.0
            if is_ref_query:
                if total > 0 and chunk_idx >= total * 0.85:
                    weight *= 0.85
                if is_ref:
                    weight *= 0.8
            weighted.append((doc, score * weight))
            print(f"  L2={score:.4f} 权重={weight:.2f} | {doc.page_content[:60].replace(chr(10), ' ')}...")

        # 按加权后分数排序
        weighted.sort(key=lambda x: x[1])
        min_score = weighted[0][1]

        # ---- 优先使用章级父块 ----
        grandparent_ids = set()
        for doc, _ in weighted:
            gp_id = doc.metadata.get("grandparent_id")
            if gp_id:
                grandparent_ids.add(gp_id)

        if grandparent_ids:
            gp_results = grandparent_store.get(
                ids=list(grandparent_ids),
                include=["documents", "metadatas"],
            )
            return_docs = [
                Document(page_content=content, metadata=meta)
                for content, meta in zip(gp_results["documents"], gp_results["metadatas"])
            ]
            return_docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
            print(f"  → 命中 {len(grandparent_ids)} 个章级父块")
            return return_docs, min_score

        # ---- 降级：节级父块 ----
        seen_parent_ids = set()
        for doc, _ in weighted:
            pid = doc.metadata.get("parent_id")
            if pid:
                seen_parent_ids.add(pid)

        if not seen_parent_ids:
            return [], 999.0

        parent_results = parent_store.get(
            ids=list(seen_parent_ids),
            include=["documents", "metadatas"],
        )
        parent_docs = [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(parent_results["documents"], parent_results["metadatas"])
        ]
        parent_docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
        print(f"  → 降级：命中 {len(seen_parent_ids)} 个节级父块")
        return parent_docs, min_score

    except Exception as e:
        print(f"[检索异常] {e}")
        return [], 999.0


def _web_answer(question: str, chat_history: list) -> str:
    """知识库无相关内容时，调用百炼联网模型回答；联网失败则降级到普通模型"""
    try:
        messages = [{"role": "system", "content": "你是一个智能助手，请联网搜索后回答用户问题，并注明信息来源。"}]
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
        messages.append({"role": "user", "content": question})

        # 先尝试联网搜索
        response = dashscope.Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qwen-plus-latest",
            messages=messages,
            extra_body={"enable_search": True},
        )
        if response.status_code == 200 and response.output and response.output.choices:
            return response.output.choices[0].message.content

        # 联网失败，降级到普通模型
        print(f"[联网搜索] 状态码={response.status_code} 错误码={response.code} 错误信息={response.message}")
        resp2 = dashscope.Generation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qwen-max",
            messages=messages,
            result_format="message",
        )
        if resp2.status_code == 200 and resp2.output and resp2.output.choices:
            return "(未找到相关论文内容，以下为通用回答)\n\n" + resp2.output.choices[0].message.content
        return f"模型调用失败: status={resp2.status_code} code={resp2.code} msg={resp2.message}"
    except Exception as e:
        return f"联网搜索出错: {str(e)}"


def build_chain():
    """构建问答链，输入需包含 question, chat_history, context"""

    system_prompt = """你是一个学术论文智能问答助手。我会提供若干知识库片段，其中部分可能与问题相关，部分可能无关。

你的任务：
1. 从提供的片段中找出与问题真正相关的内容
2. 基于相关内容给出准确回答
3. 如果问题涉及参考文献数量，请仔细数出所有编号（如[1][2]...或1. 2.），给出准确总数
4. 如果问题涉及某个主题引用了哪些文献，列出具体文献条目
5. 如果所有片段都与问题无关，直接说"未在知识库中找到相关信息"
6. 不要编造知识库中没有的内容

知识库片段：
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": RunnablePassthrough() | RunnableLambda(lambda x: x.get("context", "（无相关信息）")),
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnablePassthrough() | RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | prompt
        | RunnableLambda(_call_qwen)
        | StrOutputParser()
    )
    return chain


def get_answer(question: str, chat_history: list) -> str:
    """统一入口（仅供兼容，实际使用 app.py 中的直接检索+链调用）"""
    docs, min_score = _retrieve(question)
    if docs and min_score < 1.5:
        context = _format_docs(docs)
        chain = build_chain()
        return chain.invoke({
            "question": question,
            "chat_history": chat_history,
            "context": context,
        })
    else:
        return _web_answer(question, chat_history)
