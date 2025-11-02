import hashlib
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils.embedding_utils import create_embeddings


class TextProcessor:
    @staticmethod
    def chunk_text_by_size(text: str, chunk_size: int, overlap_size: int) -> List[str]:
        """固定大小分块"""
        text_chunks = []
        for i in range(0, len(text), chunk_size - overlap_size):
            text_chunks.append(text[i:i + chunk_size])
        return text_chunks

    @staticmethod
    def chunk_text_by_semantics(text: str, method: str = "percentile", threshold: int = 90) -> List[str]:
        """语义分块"""
        sentences = text.split("。")
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if not sentences:
            return []

        # 如果句子太少（<=1），无法计算相邻句子的相似度，直接将全文作为一个块返回
        if len(sentences) <= 1:
            return [text]

        embeddings = create_embeddings(sentences)

        # 如果嵌入生成失败或返回的嵌入数量不足，退回到整段作为单个块
        if not embeddings or len(embeddings) < 2:
            return [text]

        similarities = [
            cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            for i in range(len(embeddings) - 1)
        ]

        # 如果相似度数组为空，也直接返回完整文本
        if not similarities:
            return [text]

        breakpoints = TextProcessor._compute_breakpoints(similarities, method, threshold)
        return TextProcessor._split_into_chunks(sentences, breakpoints)

    @staticmethod
    def _compute_breakpoints(similarity_scores: List[float], method: str = "percentile", threshold: int = 90) -> List[
        int]:
        """计算语义分块的断点"""
        if method == "percentile":
            threshold_value = np.percentile(similarity_scores, threshold)
        elif method == "standard_deviation":
            mean = np.mean(similarity_scores)
            std_dev = np.std(similarity_scores)
            threshold_value = mean - (threshold * std_dev)
        elif method == "interquartile":
            q1, q3 = np.percentile(similarity_scores, [25, 75])
            iqr = q3 - q1
            threshold_value = q1 - 1.5 * iqr
        else:
            raise ValueError("无效的断点计算方法！可选：percentile/standard_deviation/interquartile")

        return [i for i, score in enumerate(similarity_scores) if score < threshold_value]

    @staticmethod
    def _split_into_chunks(sentence_list: List[str], break_indices: List[int]) -> List[str]:
        """根据断点将句子分割成块"""
        semantic_chunks = []
        current_start_index = 0

        for bp in break_indices:
            chunk = "。".join(sentence_list[current_start_index:bp + 1]) + "。"
            semantic_chunks.append(chunk)
            current_start_index = bp + 1

        if current_start_index < len(sentence_list):
            semantic_chunks.append("。".join(sentence_list[current_start_index:]) + "。")

        return semantic_chunks

    @staticmethod
    def generate_questions_from_chunk(chunk: str, num_questions: int = 3) -> List[str]:
        """从文本块生成相关问题"""
        sentences = chunk.split("。")[:3]
        questions = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                questions.append(f"关于以下内容的关键信息是什么？: {sentence.strip()}")
                if len(questions) >= num_questions:
                    break
        return questions or [f"这段文本的主要内容是什么？"]

    @staticmethod
    def chunk_by_config(text: str, config: Dict) -> List[str]:
        """根据配置选择分块方法"""
        chunk_method = config.get("chunk_method", "semantic")

        if chunk_method == "fixed_size":
            return TextProcessor.chunk_text_by_size(
                text,
                chunk_size=config.get("chunk_size", 512),
                overlap_size=config.get("overlap_size", 50)
            )
        elif chunk_method == "semantic":
            semantic_params = config.get("semantic_chunk_params", {})
            return TextProcessor.chunk_text_by_semantics(
                text,
                method=semantic_params.get("method", "percentile"),
                threshold=semantic_params.get("threshold", 90)
            )
        else:
            return TextProcessor.chunk_text_by_size(text, 512, 50)
