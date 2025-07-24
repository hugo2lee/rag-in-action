#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黑神话：悟空文档语义分块RAG系统 (Jieba增强版)
使用 llama_index 和 jieba 进行中文文档的高质量语义分块
"""

import os
import re
from typing import List, Dict, Any
from pathlib import Path

# 核心依赖
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode
from llama_index.core.text_splitter import SentenceSplitter

# 嵌入模型 - 推荐使用中文优化的模型
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# 或者使用 OpenAI 嵌入（如果有 API key）
# from llama_index.embeddings.openai import OpenAIEmbedding

# LLM 支持
from llama_index.llms.ollama import Ollama
# 或者使用 OpenAI（如果有 API key）
# from llama_index.llms.openai import OpenAI

# ----------------- 新增依赖 -----------------
# 导入Jieba用于中文分词
import jieba


# -------------------------------------------

# ----------------- 新增功能 -----------------
def jieba_tokenizer(text: str) -> List[str]:
    """
    Jieba分词器函数，用于LlamaIndex的SentenceSplitter
    """
    return list(jieba.cut(text))


# -------------------------------------------


class ChineseDocumentProcessor:
    """中文文档处理器，专门用于《黑神话：悟空》类型的游戏文档"""

    def __init__(self,
                 embedding_model_name: str = "BAAI/bge-large-zh-v1.5",
                 llm_model: str = "qwen3:0.6b",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        初始化文档处理器

        Args:
            embedding_model_name: 嵌入模型名称，推荐中文优化模型
            llm_model: LLM模型名称
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # ----------------- 新增功能：初始化Jieba并加载自定义词典 -----------------
        self._initialize_jieba()
        # ----------------------------------------------------------------------

        # 设置嵌入模型 - 使用中文优化的BGE模型
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            trust_remote_code=True
        )

        # 设置LLM - 使用Ollama运行的中文模型
        self.llm = Ollama(model=llm_model, request_timeout=120.0)

        # 配置全局设置
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap

    def _initialize_jieba(self):
        """
        初始化Jieba分词器并加载《黑神话：悟空》的自定义词典。
        这可以确保游戏中的专有名词、角色、地点等不被错误切分。
        """
        print("正在初始化Jieba并加载自定义词典...")

        # 定义游戏术语和专有名词
        game_terms = [
            '黑神话：悟空', '游戏科学', '天命人', '灵明石猴', '花果山', '如意金箍棒',
            '劈棍', '戳棍', '立棍', '定身', '聚形散气', '身外身法', '精魄', '土地庙',
            '再入轮回', '大圣残躯', '石中境', '六根', '根器', '紧箍咒', '紧箍儿',
            # 角色
            '孙悟空', '二郎神', '杨戬', '猪八戒', '天蓬元帅', '铁扇公主', '红孩儿', '牛魔王',
            '蜘蛛精', '紫蛛儿', '百眼魔君', '黄风大圣', '黑熊精', '亢金龙', '黄眉大王',
            '弥勒菩萨', '灵吉菩萨', '老猴子', '袁守诚', '无头僧',
            # 地点
            '黑风山', '观音禅院', '黄风岭', '小西天', '盘丝岭', '盘丝洞', '火焰山', '梅山',
            '大足石刻', '南禅寺', '广胜寺', '鹳雀楼',
            # 其他
            '虚幻引擎5', '3A游戏', 'WeGame', '云宫迅音', '敢问路在何方', '八零八二音频'
        ]

        for term in game_terms:
            jieba.add_word(term,1)

        print(f"Jieba自定义词典加载完成，共添加 {len(game_terms)} 个词。")

    def preprocess_chinese_text(self, text: str) -> str:
        """
        预处理中文文本，优化分块效果

        Args:
            text: 原始文本

        Returns:
            处理后的文本
        """
        # 处理章节标题，确保它们成为独立的语义单元
        text = re.sub(r'^([第\d一二三四五六七八九十]+[章节部分].*?)$', r'\n\1\n', text, flags=re.MULTILINE)

        # 清理多余空格和换行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def semantic_chunking(self, text: str) -> List[BaseNode]:
        """
        使用语义分块器进行智能分块

        Args:
            text: 文档文本

        Returns:
            分块后的节点列表
        """
        # 预处理文本
        processed_text = self.preprocess_chinese_text(text)

        # 创建文档对象
        document = Document(
            text=processed_text,
            metadata={
                'source': '黑神话悟空wiki.txt',
                'language': 'chinese',
                'domain': 'gaming'
            }
        )

        # 使用语义分块器
        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=3,
            breakpoint_percentile_threshold=50,
            embed_model=self.embed_model
        )

        # 备用：句子分块器作为后备
        # ----------------- 核心改进：在此处应用Jieba分词器 -----------------
        sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            tokenizer=jieba_tokenizer,  # 使用Jieba进行分词
            paragraph_separator="\n\n",
            secondary_chunking_regex="[。！？；]",
        )
        # ----------------------------------------------------------------------

        try:
            # 尝试语义分块
            print("正在尝试语义分块...")
            nodes = semantic_splitter.get_nodes_from_documents([document])
            print(f"语义分块完成，生成 {len(nodes)} 个节点。")
        except Exception as e:
            print(f"语义分块失败，回退至句子分块器: {e}")
            # 如果语义分块失败，使用集成了Jieba的句子分块器
            nodes = sentence_splitter.get_nodes_from_documents([document])
            print(f"句子分块完成，生成 {len(nodes)} 个节点。")

        return nodes

    def enhance_node_metadata(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """
        增强节点元数据，提高检索质量

        Args:
            nodes: 原始节点列表

        Returns:
            增强后的节点列表
        """
        enhanced_nodes = []
        print("正在增强节点元数据...")

        for i, node in enumerate(nodes):
            content = node.text

            # 检测内容类型
            content_type = self._detect_content_type(content)

            # 提取关键实体
            entities = self._extract_game_entities(content)

            # 更新元数据
            node.metadata.update({
                'chunk_id': i,
                'content_type': content_type,
                'entities': entities,
                'char_count': len(content),
                # 使用jieba分词结果估计token数，更准确
                'estimated_tokens': len(jieba_tokenizer(content)),
            })

            enhanced_nodes.append(node)

        print("节点元数据增强完成。")
        return enhanced_nodes

    def _detect_content_type(self, content: str) -> str:
        """检测内容类型"""
        if any(keyword in content for keyword in ['玩法', '战斗', '技能', '操作', '棍法', '奇术', '身法']):
            return 'gameplay'
        elif any(keyword in content for keyword in ['故事', '情节', '角色', '剧情', '结局', '设定']):
            return 'story'
        elif any(keyword in content for keyword in ['开发', '制作', '技术', '引擎', '取景', '团队']):
            return 'development'
        elif any(keyword in content for keyword in ['销量', '反响', '评价', '争议', 'Metacritic', 'IGN']):
            return 'reception'
        else:
            return 'general'

    def _extract_game_entities(self, content: str) -> List[str]:
        """提取游戏相关实体"""
        entities = []

        all_entities = [
            '孙悟空', '天命人', '二郎神', '猪八戒', '铁扇公主', '红孩儿', '牛魔王',
            '蜘蛛精', '黄风大圣', '弥勒菩萨', '灵吉菩萨', '黑熊精', '花果山',
            '火焰山', '盘丝洞', '小西天', '黑风山', '黄风岭', '六根', '根器',
            '紧箍咒', '如意金箍棒', '奇术', '身法', '毫毛', '变化'
        ]

        for entity in all_entities:
            if entity in content:
                entities.append(entity)

        return list(set(entities))

    def build_vector_index(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """
        构建向量索引

        Args:
            nodes: 文档节点列表

        Returns:
            向量索引对象
        """
        print("正在构建向量索引...")
        index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        print("向量索引构建完成。")
        return index

    def process_document(self, file_path: str) -> VectorStoreIndex:
        """
        处理整个文档的主流程

        Args:
            file_path: 文档文件路径

        Returns:
            构建好的向量索引
        """
        print(f"开始处理文档: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文档未找到: {file_path}")

        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"文档长度: {len(text)} 字符")

        # 语义分块
        nodes = self.semantic_chunking(text)

        # 增强元数据
        enhanced_nodes = self.enhance_node_metadata(nodes)

        # 构建索引
        index = self.build_vector_index(enhanced_nodes)

        # 保存分块信息（可选，用于调试）
        self.save_chunks_info(enhanced_nodes)

        print("文档处理全部流程完成！")
        return index

    def save_chunks_info(self, nodes: List[BaseNode], output_path: str = "chunks_info_jieba.txt"):
        """
        保存分块信息到文件，便于调试和分析

        Args:
            nodes: 节点列表
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, node in enumerate(nodes):
                f.write(f"================ Chunk {i + 1} ================\n")
                f.write(f"ID: {node.id_}\n")
                f.write(f"Content Type: {node.metadata.get('content_type', 'unknown')}\n")
                f.write(f"Entities: {', '.join(node.metadata.get('entities', []))}\n")
                f.write(f"Character Count: {node.metadata.get('char_count', 0)}\n")
                f.write(f"Estimated Tokens (Jieba): {node.metadata.get('estimated_tokens', 0)}\n")
                f.write("---------- Content ----------\n")
                f.write(f"{node.text}\n\n")

        print(f"分块信息已保存到: {output_path}")


def main():
    """主函数示例"""

    # 确保文档路径正确，这里假设脚本和文档在同一目录下或使用相对路径
    # 请根据您的文件结构修改此路径
    doc_path = "../90-文档-Data/黑悟空/黑悟空wiki.txt"

    # 检查文件是否存在
    if not os.path.exists(doc_path):
        print(f"错误：找不到文档文件 '{doc_path}'。请确保文件存在于正确的路径。")
        return

    # 创建处理器实例
    processor = ChineseDocumentProcessor(
        embedding_model_name="BAAI/bge-large-zh-v1.5",  # 中文优化的嵌入模型
        llm_model="qwen3:0.6b",  # Ollama中性能较好的中文LLM
        chunk_size=512,
        chunk_overlap=50
    )

    try:
        # 处理文档并构建索引
        index = processor.process_document(doc_path)

        # 创建查询引擎
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )

        # 示例查询
        test_queries = [
            "黑神话悟空的主要玩法是什么？请详细说明。",
            "游戏中的六根指的是哪六根？它们是如何被找回的？",
            "游戏的故事结局是怎样的？孙悟空最后复活了吗？",
            "这款游戏的销量和市场反响如何？",
            "游戏开发团队是哪个公司？他们为了制作游戏在哪些地方进行了实地取景？"
        ]

        print("\n================= RAG 系统查询测试 =================\n")
        for query in test_queries:
            print(f"❓ 问题: {query}")
            response = query_engine.query(query)
            print(f"🤖 回答: {response}")
            print("-" * 70)

        # 保存索引以备后用（可选）
        index.storage_context.persist(persist_dir="./wukong_index_jieba")
        print("索引已保存到 ./wukong_index_jieba 目录")

    except Exception as e:
        print(f"处理过程中出现严重错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 确保运行main函数前，你已经安装了所有必需的库：
    # pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-huggingface jieba
    # 并且已经通过Ollama运行了Qwen2模型: ollama run qwen2:1.5b
    main()