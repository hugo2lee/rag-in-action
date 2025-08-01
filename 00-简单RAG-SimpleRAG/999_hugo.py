from sentence_transformers import SentenceTransformer
import numpy as np

# 1. 加载 BAAI 嵌入模型
model_name = "BAAI/bge-large-zh-v1.5"  # 中文优化的 bge 模型
model = SentenceTransformer(model_name)

# 2. 文档分段（云冈石窟示例）
documents = [
    "云冈石窟位于山西省大同市西郊17公里处武周山南麓，东西绵延1公里，拥有45个主要洞窟，252个窟龛，51000余尊石雕造像，是中国四大石窟艺术宝库之一。1961年被列为重点文物保护单位，2001年被列入世界遗产，2007年评为国家5A级旅游景区。",
    "云冈五华洞位于第9-13窟，因清代彩绘而得名，雕饰丰富，是研究北魏历史和艺术的重要资料。",
    "塔洞位于东部窟群第1-4窟，雕有塔柱和多佛像，第3窟为最大洞窟，传为昙曜译经楼。",
    "武周山别名武州山，是云冈石窟开凿的山体，南面壁立千仞。",
    "昙曜五窟位于第16-20窟，最早开凿，主像释迦、高15.6米三世佛等，壁雕千佛。",
    "云冈石窟最佳旅游时间5月-10月，开放时间8:30-17:30（旺季），门票旺季125元，淡季80元，交通可乘坐公交或出租车。"
]

# 3. 生成文档嵌入
document_embeddings = model.encode(documents, normalize_embeddings=True)

# 4. 定义余弦相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)

# 5. 搜索函数
def search(query, top_k=3):
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = [cosine_similarity(query_embedding, emb) for emb in document_embeddings]
    ranked_results = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
    return ranked_results[:top_k]

# 6. 测试查询
query = "云冈石窟有什么价值"
results = search(query)

print(f"🔍 查询: {query}")
for i, (doc, score) in enumerate(results):
    print(f"\nTop {i+1} (相似度 {score:.4f}):\n{doc}")
