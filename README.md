# RAG 学术论文知识库问答系统

基于 LangChain + ChromaDB + 阿里云百炼的智能问答系统，支持 PDF 论文上传、三层分级检索、联网搜索兜底。

## 功能

- 💬 **智能问答**：基于知识库的学术论文内容回答问题
- 📤 **文件上传**：支持 PDF / TXT / CSV / JSON 文件向量化入库
- 🌐 **联网搜索**：知识库未命中时自动联网搜索
- 📚 **知识库管理**：查看/删除已索引的文件

## 技术栈

| 层级 | 技术 |
|------|------|
| UI | Streamlit |
| 编排 | LangChain |
| 向量库 | ChromaDB |
| 嵌入 | 阿里百炼 text-embedding-v4 |
| 问答 | 阿里百炼 qwen-max |
| PDF解析 | MinerU API |

## 本地运行

```bash
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
streamlit run app.py
```

## 环境变量

创建 `.env` 文件或 `.streamlit/secrets.toml`：

```
DASHSCOPE_API_KEY=你的百炼API_KEY
MINERU_API_TOKEN=你的MinerU_TOKEN
```
