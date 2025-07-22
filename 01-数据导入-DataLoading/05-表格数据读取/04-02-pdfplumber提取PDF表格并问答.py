import pdfplumber
import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from typing import List

pdf_path = "../../90-文档-Data/复杂PDF/billionaires_page-1-5.pdf"

# 打开 PDF 并解析表格
with pdfplumber.open(pdf_path) as pdf:
    tables = []
    for page in pdf.pages:
        table = page.extract_table()
        if table:
            tables.append(table)

# 转换所有表格为 DataFrame 并构建文档
documents: List[Document] = []
if tables:
    # 遍历所有表格
    for i, table in enumerate(tables, 1):
        # 将表格转换为 DataFrame
        df = pd.DataFrame(table)
        
        # 保存到CSV文件
        csv_filename = f"billionaires_table_{i}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n表格 {i} 数据已保存到 {csv_filename}")
        
        # 将DataFrame转换为文本
        text = df.to_string()
        
        # 创建Document对象
        doc = Document(text=text, metadata={"source": f"表格{i}"})
        documents.append(doc)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")


# 构建索引
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

from llama_index.llms.ollama import Ollama

# 初始化 Ollama LLM
ollama_llm = Ollama(model="qwen3:0.6b") # 将 "your_ollama_model_name" 替换为你的 Ollama 模型名称

# 创建查询引擎
query_engine = index.as_query_engine(llm=ollama_llm)

# 示例问答
questions = [
    "2023年谁是最富有的人?",
    "最年轻与最年长的富豪是谁?"
]

print("\n===== 问答演示 =====")
for question in questions:
    response = query_engine.query(question)
    print(f"\n问题: {question}")
    print(f"回答: {response}")