from FlagEmbedding import BGEM3FlagModel

def main():
    # 加载模型
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    # 输入文本
    passage = ["猢狲施展烈焰拳，击退妖怪；随后开启金刚体，抵挡神兵攻击。"]

    # 编码文本，获取三种嵌入
    embeddings = model.encode(
        passage,
        return_sparse=True,       # 稀疏嵌入
        return_dense=True,        # 密集嵌入
        return_colbert_vecs=True  # 多向量嵌入
    )

    # 分别提取三种嵌入
    dense_vecs = embeddings["dense_vecs"]        # (1, 1024)
    sparse_vecs = embeddings["lexical_weights"]  # List[Dict[token_id -> weight]]
    colbert_vecs = embeddings["colbert_vecs"]    # List of arrays (tokens × 1024)

    # 1. 输出 Dense embedding 信息
    print("\n===== Dense Embedding =====")
    print("密集嵌入维度:", dense_vecs[0].shape)
    print("前10维:", dense_vecs[0][:10])

    # 2. 输出 Sparse embedding 信息（带中文 token）
    print("\n===== Sparse Embedding =====")
    sparse_dict = sparse_vecs[0]  # 当前文本的稀疏向量
    print("稀疏嵌入非零元素个数:", len(sparse_dict))

    # 将 token_id 转换成对应 token
    token_ids = [int(k) for k in sparse_dict.keys()]
    tokens = model.tokenizer.convert_ids_to_tokens(token_ids)
    weights = [float(v) for v in sparse_dict.values()]

    print("前10个 token 及其权重:")
    for t, w in list(zip(tokens, weights))[:10]:
        print(f"{t}: {w:.4f}")

    # 构建 ElasticSearch 可用的格式
    # 示例格式: {"猢": 0.12, "狲": 0.09, ...}
    sparse_for_es = dict(zip(tokens, weights))
    print("\nElasticSearch 稀疏向量示例:")
    print(sparse_for_es)

    # 3. 输出 Multi-vector embedding 信息
    print("\n===== Multi-vector Embedding (ColBERT) =====")
    print("多向量嵌入维度:", colbert_vecs[0].shape)
    print("前2个 token 向量:")
    print(colbert_vecs[0][:2])  # 显示前两个 token 的向量

if __name__ == '__main__':
    main()
