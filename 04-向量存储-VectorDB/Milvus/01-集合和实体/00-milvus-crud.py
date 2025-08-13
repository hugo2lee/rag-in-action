"""
Author: hugo2lee hugo2lee@gmail.com
Date: 2025-08-12 14:20:22
LastEditors: hugo2lee hugo2lee@gmail.com
LastEditTime: 2025-08-13 10:51:39
FilePath: /rag-in-action/04-向量存储-VectorDB/Milvus/01-集合和实体/00-milvus-crud.py
Description:
"""

# -*- coding: utf-8 -*-
from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")

# if client.has_collection(collection_name="demo_collection"):
#     client.drop_collection(collection_name="demo_collection")
# client.create_collection(
#     collection_name="demo_collection",
#     dimension=768,  # The vectors we will use in this demo has 768 dimensions
# )

from pymilvus import model


embedding_fn = model.DefaultEmbeddingFunction()

# docs = [
#     "Artificial intelligence was founded as an academic discipline in 1956.",
#     "Alan Turing was the first person to conduct substantial research in AI.",
#     "Born in Maida Vale, London, Turing was raised in southern England.",
# ]

# vectors = embedding_fn.encode_documents(docs)
# print("Dim:", embedding_fn.dim, vectors[0].shape)  # Dim: 768 (768,)

# data = [
#     {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
#     for i in range(len(vectors))
# ]

# print("Data has", len(data), "entities, each with fields: ", data[0].keys())
# print("Vector dim:", len(data[0]["vector"]))

# res = client.insert(collection_name="demo_collection", data=data)

# print(res)

query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print("Who is Alan Turing? result:", res)

# docs = [
#     "Machine learning has been used for drug design.",
#     "Computational synthesis with AI algorithms predicts molecular properties.",
#     "DDR1 is involved in cancers and fibrosis.",
# ]
# vectors = embedding_fn.encode_documents(docs)
# data = [
#     {"id": 3 + i, "vector": vectors[i], "text": docs[i], "subject": "biology"}
#     for i in range(len(vectors))
# ]

# client.insert(collection_name="demo_collection", data=data)

res = client.search(
    collection_name="demo_collection",
    data=embedding_fn.encode_queries(["tell me AI related information"]),
    filter="subject == 'biology'",
    limit=2,
    output_fields=["text", "subject"],
)

print("tell me AI related information result:", res)

# from pymilvus import model
# import numpy as np

# embedding_fn = model.DefaultEmbeddingFunction()

# query = "Who is Alan Turing?"
# doc1 = "Alan Turing was the first person to conduct substantial research in AI."
# doc2 = "Born in Maida Vale, London, Turing was raised in southern England."

# # 编码
# query_vec = embedding_fn.encode_queries([query])[0]
# doc1_vec = embedding_fn.encode_documents([doc1])[0]
# doc2_vec = embedding_fn.encode_documents([doc2])[0]


# # 余弦相似度函数
# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# print("doc1 相似度:", cosine_similarity(query_vec, doc1_vec))
# print("doc2 相似度:", cosine_similarity(query_vec, doc2_vec))

# # 欧几里得距离
# print("doc1 距离:", np.linalg.norm(doc1_vec - query_vec))
# print("doc2 距离:", np.linalg.norm(doc2_vec - query_vec))

# # 曼哈顿距离
# print("doc1 曼哈顿距离:", np.sum(np.abs(doc1_vec - query_vec)))
# print("doc2 曼哈顿距离:", np.sum(np.abs(doc2_vec - query_vec)))

res = client.query(
    collection_name="demo_collection",
    filter="subject == 'history'",
    output_fields=["text", "subject"],
)
print("query history result:", res)

res = client.query(
    collection_name="demo_collection",
    ids=[0, 2],
    output_fields=["vector", "text", "subject"],
)
print("query ids result:", res)
