import fitz

doc = fitz.open("d:/学习/RAG学习/城市社区尺度下空间对儿童健康的作用机理研究进展_裴昱.pdf")
for i, page in enumerate(doc, start=1):
    text = page.get_text("text")
    print(f"\n=== 第{i}页，字符数: {len(text)} ===")
    print(text[:300])
doc.close()