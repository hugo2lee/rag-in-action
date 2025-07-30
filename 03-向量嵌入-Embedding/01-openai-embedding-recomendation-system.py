import os
import openai
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('BAAI/bge-large-en-v1.5')


# 读取用户评价数据集
df = pd.read_csv("../90-文档-Data/灭神纪/用户评价.csv")

# 读取游戏描述文件
with open("../90-文档-Data/灭神纪/游戏说明.json", "r") as f:
    game_descriptions = json.load(f)

# 定义函数获取嵌入向量
# def get_embedding(text, model="text-embedding-3-small"):
#     response = openai.embeddings.create(
#         input=[text],
#         model=model
#     )
#     return response.data[0].embedding

def get_embedding(text):
    return model.encode(text, normalize_embeddings=True)  # normalize_embeddings=True 让向量归一化，适合余弦相似度

# 获取所有游戏的嵌入向量
unique_games = df['game_title'].unique().tolist()
target_game = "Killing God: Hu Sun"  # 目标游戏名称更改
if target_game not in unique_games:
    unique_games.append(target_game)  # 确保目标游戏在列表中
game_embeddings = {}
for game in unique_games:
    description = game_descriptions[game]
    game_embeddings[game] = np.array(get_embedding(description))

# 计算用户评价的嵌入向量（该用户评价过的所有游戏描述嵌入向量的平均值）
user_vectors = {}
for user_id, group in df.groupby("user_id"):
    user_game_vecs = []
    for idx, row in group.iterrows():
        g_title = row['game_title']
        g_vec = game_embeddings[g_title]
        user_game_vecs.append(g_vec)
    user_vectors[user_id] = np.mean(np.array(user_game_vecs), axis=0)

# 获取“灭神纪·猢狲”的嵌入向量
target_vector = game_embeddings[target_game]
# 计算每个用户评价的嵌入向量与目标游戏的嵌入向量的余弦相似度
results = []
for user_id, u_vec in user_vectors.items():
    u_vec_reshaped = u_vec.reshape(1, -1)
    t_vec = target_vector.reshape(1, -1)
    similarity = cosine_similarity(u_vec_reshaped, t_vec)[0,0]
    results.append((user_id, similarity))

# 排序并找出最可能喜欢“灭神纪·猢狲”的用户
result_df = pd.DataFrame(results, columns=["user_id", f"similarity_to_{target_game}"])
result_df = result_df.sort_values(by=f"similarity_to_{target_game}", ascending=False)
print(f"\n最可能喜欢{target_game}的前5位用户：")
print(result_df.head())






import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 加载模型
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# 2. 读取用户评价数据
df = pd.read_csv("../90-文档-Data/灭神纪/用户评价.csv")

# 3. 读取游戏描述
with open("../90-文档-Data/灭神纪/游戏说明.json", "r") as f:
    game_descriptions = json.load(f)

# 4. 目标用户 & 游戏
target_user = "U003"
target_game = "Killing God: Hu Sun"
target_desc = game_descriptions[target_game]

# 5. 获取 U003 玩过的游戏列表
user_games = df[df['user_id'] == target_user]['game_title'].tolist()
print(f"U003 玩过的游戏：{user_games}")

# 6. 计算目标游戏嵌入
target_vec = model.encode(target_desc, normalize_embeddings=True)

# 7. 计算每个用户游戏与目标游戏的相似度
similarities = []
for game in user_games:
    if game in game_descriptions:
        desc = game_descriptions[game]
        game_vec = model.encode(desc, normalize_embeddings=True)
        sim = np.dot(game_vec, target_vec)  # 直接点积，因为已归一化
        similarities.append((game, sim))

# 8. 排序并显示
similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

print(f"\nU003 的每款游戏与 {target_game} 的相似度：")
for g, s in similarities:
    print(f"{g:<30} {s:.4f}")

# 9. 解释：哪些关键词贡献相似性
print("\n关键原因：")
print("- 所有游戏都是 Action RPG，战斗元素强。")
print("- Nioh、Sekiro 与目标游戏有东方文化背景。")
print("- Elden Ring 与 Hu Sun 都强调开放世界和神话元素。")
