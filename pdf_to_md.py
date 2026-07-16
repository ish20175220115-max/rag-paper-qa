import mineru
print(dir(mineru))


converter = MinerU()
result = converter.convert("d:/学习/RAG学习/城市社区尺度下空间对儿童健康的作用机理研究进展_裴昱.pdf")
md_text = result.markdown

# 保存到文件
with open("d:/学习/RAG学习/output.md", "w", encoding="utf-8") as f:
    f.write(md_text)

print(f"转换完成，Markdown 长度: {len(md_text)} 字符")
print(md_text[:500])
