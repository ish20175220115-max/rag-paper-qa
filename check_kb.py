from knowledge import get_vector_store

vs = get_vector_store()
all_data = vs._collection.get(include=["documents", "metadatas"])

docs = list(zip(all_data["documents"], all_data["metadatas"]))
print(f"向量库总块数: {len(docs)}")
print("="*60)

# 按文件分组显示
from collections import defaultdict
file_chunks = defaultdict(list)
for content, meta in docs:
    source = meta.get("source", "未知")
    chunk_idx = meta.get("chunk_index", -1)
    file_chunks[source].append((chunk_idx, content))

for source, chunks in file_chunks.items():
    chunks.sort(key=lambda x: x[0])
    print(f"\n📄 文件: {source}")
    print(f"   块数: {len(chunks)}")
    print("-"*60)
    for idx, content in chunks:
        print(f"  块{idx+1:3d} | 长度{len(content):4d} | {content[:100].replace(chr(10), ' ')}...")
