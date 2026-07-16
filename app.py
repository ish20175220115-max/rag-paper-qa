from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import tempfile
import uuid
from knowledge import add_files, list_files, delete_file
from rag import build_chain, _retrieve, _format_docs, _web_answer
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="RAG 知识库问答", layout="wide")

# ---------- session_state ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = build_chain()  # 仅构建一次
if "page" not in st.session_state:
    st.session_state.page = "qa"

# ---------- 侧边栏 ----------
with st.sidebar:
    st.markdown("## 🧭 导航")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 问答", use_container_width=True):
            st.session_state.page = "qa"
            st.rerun()
    with col2:
        if st.button("📤 上传", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()

    st.divider()

    st.markdown("### 📚 知识库管理")
    files = list_files()
    if files:
        st.caption(f"共 {len(files)} 个文件")
        for fname in files:
            col_f, col_d = st.columns([3, 1])
            with col_f:
                st.text(fname)
            with col_d:
                if st.button("🗑", key=f"del_{fname}"):
                    deleted = delete_file(fname)
                    if deleted > 0:
                        st.success(f"已删除 {deleted} 个块")
                        st.rerun()
                    else:
                        st.warning(f"未找到文件 {fname}")
    else:
        st.caption("知识库为空")

    st.divider()

    file_count = len(list_files())
    st.info(f"📚 当前索引文件数: {file_count}")

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ---------- 问答页面 ----------
if st.session_state.page == "qa":
    st.title("🤖 RAG 智能问答")
    st.caption("基于本地知识库，未命中时自动联网搜索")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("请输入问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("检索中..."):
                try:
                    # ---- 预先检索，获得 docs 和 min_score ----
                    docs, min_score = _retrieve(prompt)
                    if docs and min_score < 1.5:
                        context = _format_docs(docs)
                        # 直接调用链，传入 context，避免重复检索
                        answer = st.session_state.chain.invoke({
                            "question": prompt,
                            "chat_history": st.session_state.chat_history,
                            "context": context,
                        })
                    else:
                        st.toast("📡 知识库未命中，正在联网搜索...")
                        answer = _web_answer(prompt, st.session_state.chat_history)
                except Exception as e:
                    answer = f"❌ 出错：{str(e)}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=answer))

        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]

# ---------- 上传页面 ----------
elif st.session_state.page == "upload":
    st.title("📁 上传文件至知识库")
    st.caption("支持 PDF / TXT / CSV / JSON")

    uploaded_files = st.file_uploader(
        "选择文件",
        type=["pdf", "txt", "csv", "json"],
        accept_multiple_files=True,
    )

    if st.button("📥 添加到知识库"):
        if not uploaded_files:
            st.warning("请先选择文件")
        else:
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            for f in uploaded_files:
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as fp:
                    fp.write(f.getbuffer())
                file_paths.append(path)

            with st.spinner("向量化中，请稍候..."):
                try:
                    results = add_files(file_paths)
                    for res in results:
                        label = f"📄 {res['file_name']}（{res['num_chunks']} 个片段）"
                        with st.expander(label):
                            if res["previews"]:
                                for i, p in enumerate(res["previews"]):
                                    st.markdown(f"**片段 {i+1}**")
                                    st.write(p["content"])
                                    st.caption(f"元数据: {p['metadata']}")
                            else:
                                st.warning("未生成有效片段，请检查文件内容")
                    st.success(f"✅ 成功添加 {len(uploaded_files)} 个文件")
                    st.rerun()
                except Exception as e:
                    st.error(f"处理失败: {str(e)}")
                finally:
                    for p in file_paths:
                        if os.path.exists(p):
                            os.remove(p)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)