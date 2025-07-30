# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# 编码文本（支持列表）
sentences = ["I love AI", "Machine learning is amazing"]
embeddings = model.encode(sentences, normalize_embeddings=True)

print(embeddings.shape)  # (2, 1024) 向量维度取决于模型

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 两个句子
s1 = "Artificial Intelligence is the future"
s2 = "AI will change the world"

vec1 = model.encode(s1, normalize_embeddings=True)
vec2 = model.encode(s2, normalize_embeddings=True)

similarity = cosine_similarity([vec1], [vec2])[0][0]
print(f"相似度: {similarity}")
