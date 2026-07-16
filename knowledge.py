import os
import re
import time
import uuid
import zipfile
import io
import requests as req
from collections import defaultdict, Counter
import fitz  # PyMuPDF（备用）
from langchain_community.document_loaders import TextLoader, CSVLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from embeddings import DashScopeEmbeddings
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_PERSIST_DIR,
    SUPPORTED_FILE_TYPES,
    MINERU_API_TOKEN,
)
import chromadb


def get_vector_store() -> Chroma:
    """子块向量库，用于检索（包含向量嵌入）"""
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=DashScopeEmbeddings(),
        collection_name="child_chunks",
    )


def get_parent_store():
    """
    节级父块存储库（纯存储，无向量索引）
    返回 chromadb Collection 对象，直接按 ID 存取完整内容
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(name="parent_chunks")


def get_grandparent_store():
    """
    章级父块存储库（纯存储，无向量索引）
    返回 chromadb Collection 对象，直接按 ID 存取完整内容
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(name="grandparent_chunks")


def _split(file_path: str):
    """
    返回 (grandparent_docs, parent_docs, child_docs)
    grandparent_docs: 章级父块（按 H2 切割的完整章）
    parent_docs: 节级父块（按 H2+H3 切割的完整节）
    child_docs: 子块（参考文献不切割，普通内容切割）
    """
    ext = file_path.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"不支持的文件类型: {ext}")

    if ext == "pdf":
        import time
        import requests as req
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        from config import MINERU_API_TOKEN

        BASE_URL = "https://mineru.net/api/v1/agent"
        file_name = os.path.basename(file_path)

        # 第一步：获取签名上传 URL
        resp = req.post(
            f"{BASE_URL}/parse/file",
            headers={"Authorization": f"Bearer {MINERU_API_TOKEN}"},
            json={
                "file_name": file_name,
                "language": "ch",
                "enable_table": True,
                "is_ocr": False,
                "enable_formula": True,
            },
        )
        result = resp.json()
        print(f"  [MinerU] 申请响应: {result}")

        if result.get("code") != 0:
            raise RuntimeError(f"MinerU 获取上传链接失败: {result.get('msg')}")

        task_id = result["data"]["task_id"]
        file_url = result["data"]["file_url"]
        print(f"  [MinerU] task_id: {task_id}")

        # 第二步：上传文件
        with open(file_path, "rb") as f:
            put_resp = req.put(file_url, data=f)
        if put_resp.status_code not in (200, 201):
            raise RuntimeError(f"MinerU 文件上传失败: {put_resp.status_code}")
        print("  [MinerU] 文件上传成功，等待解析...")

        # 第三步：轮询结果
        md_text = None
        for elapsed in range(0, 300, 3):
            time.sleep(3)
            poll_resp = req.get(
                f"{BASE_URL}/parse/{task_id}",
                headers={"Authorization": f"Bearer {MINERU_API_TOKEN}"},
            )
            poll_data = poll_resp.json()
            state = poll_data.get("data", {}).get("state")
            print(f"  [MinerU] [{elapsed}s] 状态: {state}")

            if state == "done":
                markdown_url = poll_data["data"]["markdown_url"]
                md_text = req.get(markdown_url).text
                break
            elif state == "failed":
                raise RuntimeError(f"MinerU 解析失败: {poll_data['data'].get('err_msg')}")

        if not md_text:
            raise RuntimeError("MinerU 解析超时")

        print(f"  [MinerU] 解析成功，Markdown 长度: {len(md_text)} 字符")

        # ===== 调试：打印 "4 小结" 附近内容 =====
        keyword = "4 小结"
        start = md_text.find(keyword)
        if start != -1:
            snippet = md_text[start:start+500]
            print(f"[调试] 找到 '{keyword}'，片段如下：\n{snippet}")
        else:
            print("[调试] 未找到 '4 小结' 关键词，打印全文末尾300字符：")
            print(md_text[-300:])
        # =======================================

        # ---- 第四步：两级切割（适配该论文的层级：## 为章，### 为节） ----
        # 章级父块：按二级标题切（##）
        chapter_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "H2")],
            strip_headers=False,
        )
        chapter_docs = chapter_splitter.split_text(md_text)
        for d in chapter_docs:
            d.metadata["source"] = file_path

        # 节级父块：按二三级标题切（## 章，### 节）
        section_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "H2"), ("###", "H3")],
            strip_headers=False,
        )
        section_docs = section_splitter.split_text(md_text)
        for d in section_docs:
            d.metadata["source"] = file_path

        # 如果节级切不出来，则回退到章级或全文
        docs = section_docs if section_docs else chapter_docs if chapter_docs else [
            Document(page_content=md_text, metadata={"source": file_path})
        ]

    elif ext == "txt":
        # TXT 文件无标题，直接作为单个块
        docs = TextLoader(file_path, encoding="utf-8").load()
        # 此时 docs 只有一个 Document，后续统一处理时无标题，无章级节级之分
        chapter_docs = []
        section_docs = docs
    elif ext == "csv":
        docs = CSVLoader(file_path).load()
        chapter_docs = []
        section_docs = docs
    elif ext == "json":
        docs = JSONLoader(file_path, jq_schema=".", text_content=False).load()
        chapter_docs = []
        section_docs = docs
    else:
        raise NotImplementedError

    if not docs:
        return [], [], []

    # ---- 建立章级映射（用于后续匹配） ----
    chapter_map = {}
    # 对于非 PDF（如 txt/csv/json），没有章级，直接留空
    if ext == "pdf" and chapter_docs:
        for ci, cdoc in enumerate(chapter_docs):
            chapter_id = str(uuid.uuid4())
            chapter_map[ci] = {
                "id": chapter_id,
                "doc": cdoc,
                "chunk_index": ci,
                "total_chunks": len(chapter_docs),
            }
    else:
        # 如果没有章级，则用节级本身作为章级（即单层）
        # 但为了简化，我们这里不生成章级，后续循环中 grandparent_id 留空
        pass

    # ---- 生成章级父块列表 ----
    grandparent_docs = []
    if ext == "pdf" and chapter_docs:
        for ci, cinfo in chapter_map.items():
            grandparent_docs.append(Document(
                page_content=cinfo["doc"].page_content,
                metadata={
                    **cinfo["doc"].metadata,
                    "parent_id": cinfo["id"],
                    "chunk_index": cinfo["chunk_index"],
                    "total_chunks": cinfo["total_chunks"],
                    "level": "chapter",
                }
            ))

    # ---- 二次切割器（用于子块） ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )

    # 参考文献正则（判断完整块是否为参考文献列表）
    ref_pattern = re.compile(r'(\[\d+\]|\d+\.\s+[A-Z]|\(\d{4}\))')

    parent_docs = []
    child_docs = []

    for i, doc in enumerate(docs):
        parent_id = str(uuid.uuid4())
        content = doc.page_content
        matches = ref_pattern.findall(content)
        is_ref = len(matches) >= 5

        # 查找对应的章级 parent_id（通过 H2 标题匹配，因为章级块使用 H2）
        doc_h2 = doc.metadata.get("H2", "")
        grandparent_id = None
        if ext == "pdf" and chapter_docs:
            for cinfo in chapter_map.values():
                if cinfo["doc"].metadata.get("H2", "") == doc_h2:
                    grandparent_id = cinfo["id"]
                    break

        # ---- 节级父块 ----
        parent_metadata = {
            **doc.metadata,
            "parent_id": parent_id,
            "grandparent_id": grandparent_id,
            "chunk_index": i,
            "total_chunks": len(docs),
            "is_reference": is_ref,
            "level": "section",
        }
        parent_doc = Document(
            page_content=content,
            metadata=parent_metadata,
        )
        parent_docs.append(parent_doc)

        # ---- 子块：参考文献不切割，普通内容切割 ----
        if is_ref:
            child = Document(
                page_content=content,
                metadata={**parent_metadata, "parent_id": parent_id}
            )
            child_docs.append(child)
        else:
            sub_chunks = splitter.split_documents([doc])
            for sub in sub_chunks:
                sub.metadata["parent_id"] = parent_id
                sub.metadata["grandparent_id"] = grandparent_id
                sub.metadata["is_reference"] = False
                sub.metadata["chunk_index"] = i
                sub.metadata["total_chunks"] = len(docs)
                sub.metadata["level"] = "section"
                child_docs.append(sub)

    # ---- 子块去重（基于前 100 字符） ----
    seen_prefix, unique_children = set(), []
    for d in child_docs:
        content = d.page_content.strip()
        prefix = content[:100]
        if len(content) >= 20 and prefix not in seen_prefix:
            seen_prefix.add(prefix)
            unique_children.append(d)

    print(f"[知识库] {os.path.basename(file_path)}: 章级父块 {len(grandparent_docs)} 个，节级父块 {len(parent_docs)} 个，子块 {len(unique_children)} 个")

    return grandparent_docs, parent_docs, unique_children


def delete_file(file_name: str) -> int:
    """从三个集合中同时删除指定文件名的所有块，返回删除的总块数"""
    child_vs = get_vector_store()
    parent_store = get_parent_store()
    grandparent_store = get_grandparent_store()

    total_deleted = 0

    # ---- 删除子块 ----
    child_data = child_vs._collection.get(include=["metadatas"])
    child_ids = [
        doc_id for doc_id, meta in zip(child_data["ids"], child_data["metadatas"])
        if os.path.basename(meta.get("source", "")) == file_name
    ]
    if child_ids:
        child_vs._collection.delete(ids=child_ids)
        total_deleted += len(child_ids)

    # ---- 删除节级父块 ----
    parent_data = parent_store.get(include=["metadatas"])
    parent_ids = [
        doc_id for doc_id, meta in zip(parent_data["ids"], parent_data["metadatas"])
        if os.path.basename(meta.get("source", "")) == file_name
    ]
    if parent_ids:
        parent_store.delete(ids=parent_ids)
        total_deleted += len(parent_ids)

    # ---- 删除章级父块 ----
    grandparent_data = grandparent_store.get(include=["metadatas"])
    grandparent_ids = [
        doc_id for doc_id, meta in zip(grandparent_data["ids"], grandparent_data["metadatas"])
        if os.path.basename(meta.get("source", "")) == file_name
    ]
    if grandparent_ids:
        grandparent_store.delete(ids=grandparent_ids)
        total_deleted += len(grandparent_ids)

    print(f"[知识库] 已删除 {file_name} 共 {total_deleted} 个块（子块 {len(child_ids)}，节级父块 {len(parent_ids)}，章级父块 {len(grandparent_ids)}）")
    return total_deleted


def add_files(file_paths: list) -> list:
    """向量化文件并同时存入三个集合（章级、节级、子块），自动删除同名旧数据"""
    child_vs = get_vector_store()
    parent_store = get_parent_store()
    grandparent_store = get_grandparent_store()
    results = []

    for fp in file_paths:
        file_name = os.path.basename(fp)
        # ---- 先删除同名旧数据，防止重复 ----
        delete_file(file_name)

        grandparent_docs, parent_docs, child_docs = _split(fp)
        if not child_docs:
            results.append({"file_name": file_name, "num_chunks": 0, "previews": []})
            continue

        # ---- 写入章级父块 ----
        for i in range(0, len(grandparent_docs), 50):
            batch = grandparent_docs[i: i + 50]
            grandparent_store.upsert(
                ids=[d.metadata["parent_id"] for d in batch],
                documents=[d.page_content for d in batch],
                metadatas=[d.metadata for d in batch],
            )

        # ---- 写入节级父块 ----
        for i in range(0, len(parent_docs), 50):
            batch = parent_docs[i: i + 50]
            parent_store.upsert(
                ids=[d.metadata["parent_id"] for d in batch],
                documents=[d.page_content for d in batch],
                metadatas=[d.metadata for d in batch],
            )

        # ---- 写入子块 ----
        for i in range(0, len(child_docs), 50):
            child_vs.add_documents(child_docs[i: i + 50])

        results.append({
            "file_name": file_name,
            "num_chunks": len(child_docs),
            "previews": [
                {
                    "content": d.page_content[:200] + ("..." if len(d.page_content) > 200 else ""),
                    "metadata": d.metadata,
                }
                for d in child_docs[:3]
            ],
        })

    return results


def list_files() -> list:
    """列出向量库中所有已索引的文件名（基于子块集合）"""
    vs = get_vector_store()
    all_data = vs._collection.get(include=["metadatas"])
    sources = set()
    for meta in all_data["metadatas"]:
        source = meta.get("source", "")
        if source:
            sources.add(os.path.basename(source))
    return sorted(list(sources))