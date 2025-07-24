#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黑神话：悟空文档语义分块RAG系统
使用 llama_index 进行中文文档的高质量语义分块
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

    def preprocess_chinese_text(self, text: str) -> str:
        """
        预处理中文文本，优化分块效果

        Args:
            text: 原始文本

        Returns:
            处理后的文本
        """
        # 规范化标点符号
        # text = re.sub(r'[，。；！？：""''（）【】《》〈〉]', lambda m: {
        #     '，': ', ', '。': '. ', '；': '; ', '！': '! ', '？': '? ',
        #     '：': ': ', '""': '"', '''': "'", '（': ' (', '）': ') ',
        #     '【': ' [', '】': '] ', '《': ' <', '》': '> ',
        #     '〈': ' <', '〉': '> '
        # }.get(m.group(), m.group()), text)

        # 处理章节标题，确保它们成为独立的语义单元
        text = re.sub(r'^([第\d一二三四五六七八九十]+[章节部分].*?)$', r'\n\1\n', text, flags=re.MULTILINE)

        # 处理游戏术语和专有名词，保持完整性
        game_terms = [
            '孙悟空', '天命人', '黑神话：悟空', '西游记', '花果山', '二郎神',
            '猪八戒', '铁扇公主', '红孩儿', '牛魔王', '蜘蛛精', '黄风大圣',
            '弥勒菩萨', '灵吉菩萨', '黑熊精', '六根', '根器', '紧箍咒',
            '如意金箍棒', '火焰山', '盘丝洞', '小西天', '黑风山'
        ]

        # 确保游戏术语不被错误分割
        for term in game_terms:
            text = text.replace(term, term.replace(' ', '_SPACE_'))

        # 清理多余空格和换行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        # 恢复术语中的空格
        text = text.replace('_SPACE_', ' ')

        return text.strip()

    def create_structured_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        创建结构化的文档块，保持语义完整性

        Args:
            text: 预处理后的文本

        Returns:
            结构化的文档块列表
        """
        chunks = []

        # 按主要章节分割
        sections = re.split(r'\n(?=[A-Z].*?(?:类型|平台|开发商|玩法|情节|游戏开发|发行|反响|争议))', text)

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # 提取章节标题
            lines = section.split('\n')
            title = lines[0].strip() if lines else f"章节_{i+1}"
            content = '\n'.join(lines[1:]).strip()

            # 如果内容太长，进一步细分
            if len(content) > self.chunk_size * 2:
                # 按子标题分割
                subsections = re.split(r'\n(?=[^\s].*?(?:\n|$))', content)

                for j, subsection in enumerate(subsections):
                    if len(subsection.strip()) < 50:  # 跳过太短的片段
                        continue

                    chunk = {
                        'content': subsection.strip(),
                        'metadata': {
                            'section': title,
                            'subsection': j + 1,
                            'chunk_type': 'subsection',
                            'source': '黑神话悟空wiki.txt'
                        }
                    }
                    chunks.append(chunk)
            else:
                chunk = {
                    'content': content,
                    'metadata': {
                        'section': title,
                        'chunk_type': 'section',
                        'source': '黑神话悟空wiki.txt'
                    }
                }
                chunks.append(chunk)

        return chunks

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
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model
        )

        # 备用：句子分块器作为后备
        sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[。！？；]",
        )

        try:
            # 尝试语义分块
            nodes = semantic_splitter.get_nodes_from_documents([document])
            print(f"语义分块完成，生成 {len(nodes)} 个节点")
        except Exception as e:
            print(f"语义分块失败，使用句子分块器: {e}")
            # 如果语义分块失败，使用句子分块器
            nodes = sentence_splitter.get_nodes_from_documents([document])
            print(f"句子分块完成，生成 {len(nodes)} 个节点")

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
                'estimated_tokens': len(content) // 4,  # 粗略估计中文token数
            })

            enhanced_nodes.append(node)

        return enhanced_nodes

    def _detect_content_type(self, content: str) -> str:
        """检测内容类型"""
        if any(keyword in content for keyword in ['玩法', '战斗', '技能', '操作']):
            return 'gameplay'
        elif any(keyword in content for keyword in ['故事', '情节', '角色', '剧情']):
            return 'story'
        elif any(keyword in content for keyword in ['开发', '制作', '技术', '引擎']):
            return 'development'
        elif any(keyword in content for keyword in ['销量', '反响', '评价', '争议']):
            return 'reception'
        else:
            return 'general'

    def _extract_game_entities(self, content: str) -> List[str]:
        """提取游戏相关实体"""
        entities = []

        # 角色名称
        characters = ['孙悟空', '天命人', '二郎神', '猪八戒', '铁扇公主', '红孩儿', 
                     '牛魔王', '蜘蛛精', '黄风大圣', '弥勒菩萨', '灵吉菩萨', '黑熊精']

        # 地名
        locations = ['花果山', '火焰山', '盘丝洞', '小西天', '黑风山', '黄风岭']

        # 游戏术语
        terms = ['六根', '根器', '紧箍咒', '如意金箍棒', '奇术', '身法', '毫毛', '变化']

        all_entities = characters + locations + terms

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
        print("向量索引构建完成")
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

        print("文档处理完成！")
        return index

    def save_chunks_info(self, nodes: List[BaseNode], output_path: str = "chunks_info.txt"):
        """
        保存分块信息到文件，便于调试和分析

        Args:
            nodes: 节点列表
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, node in enumerate(nodes):
                f.write(f"=== Chunk {i+1} ===\n")
                f.write(f"Content Type: {node.metadata.get('content_type', 'unknown')}\n")
                f.write(f"Entities: {', '.join(node.metadata.get('entities', []))}\n")
                f.write(f"Character Count: {node.metadata.get('char_count', 0)}\n")
                f.write(f"Content:\n{node.text}\n\n")

        print(f"分块信息已保存到: {output_path}")


def main():
    """主函数示例"""

    # 文档路径
    doc_path = "../90-文档-Data/黑悟空/黑悟空wiki.txt"

    # 创建处理器实例
    processor = ChineseDocumentProcessor(
        embedding_model_name="BAAI/bge-large-zh-v1.5",  # 中文优化的嵌入模型
        llm_model="qwen3:0.6b",  # 中文LLM模型
        chunk_size=512,
        chunk_overlap=50
    )

    try:
        # 处理文档
        index = processor.process_document(doc_path)

        # 保存分块信息（可选）
        processor.save_chunks_info(index.docstore.docs.values())

        # 创建查询引擎
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )

        # 示例查询
        test_queries = [
            "黑神话悟空的主要玩法是什么？",
            "游戏中的六根指的是什么？",
            "孙悟空在游戏中发生了什么？",
            "游戏的销量如何？",
            "游戏开发团队是谁？"
        ]

        # print("\n=== 测试查询 ===")
        # for query in test_queries:
        #     print(f"\n问题: {query}")
        #     response = query_engine.query(query)
        #     print(f"回答: {response}")
        #     print("-" * 50)

        # 保存索引（可选）
        index.storage_context.persist(persist_dir="./wukong_index")
        print("索引已保存到 ./wukong_index 目录")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()