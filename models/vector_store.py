import os
import json
import hashlib
from collections import defaultdict
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os


class VectorStore:
    def __init__(self):
        self.text_chunks: List[str] = []
        self.text_embeddings: List[List[float]] = []
        self.question_embeddings: List[List[float]] = []
        self.chunk_to_questions: Dict[int, List[int]] = defaultdict(list)
        self.question_to_chunk: Dict[int, int] = {}
        self.chunk_md5_map: Dict[int, str] = {}
        self.metadata: Dict[int, Dict] = {}
        self.chat_history: List[Dict] = []
        # (FAISS 已被移除) 保持基本内存结构
        self.faiss_index = None
        self.faiss_dim = None

    def add_chunk(self, chunk: str, source_file: str = "", questions: List[str] = None):
        """添加文本块到向量存储"""
        from utils.embedding_utils import create_embeddings
        from utils.text_processor import TextProcessor

        if not questions:
            questions = TextProcessor.generate_questions_from_chunk(chunk, num_questions=2)

        # 创建文本块的嵌入向量
        chunk_embedding = create_embeddings(chunk)[0]
        chunk_idx = len(self.text_chunks)

        # 存储文本块和嵌入向量
        self.text_chunks.append(chunk)
        self.text_embeddings.append(chunk_embedding)
        self.chunk_md5_map[chunk_idx] = hashlib.md5(chunk.encode("utf-8")).hexdigest()

        # 存储元数据
        self.metadata[chunk_idx] = {
            "source_file": source_file,
            "chunk_index": chunk_idx,
            "length": len(chunk),
            "created_at": str(np.datetime64('now'))
        }

        # 为每个问题创建嵌入向量并建立映射关系
        for question in questions:
            question_embedding = create_embeddings(question)[0]
            q_idx = len(self.question_embeddings)
            self.question_embeddings.append(question_embedding)
            self.chunk_to_questions[chunk_idx].append(q_idx)
            self.question_to_chunk[q_idx] = chunk_idx

        pass

    # ---------------- Feedback ----------------
    def apply_feedback(self, chunk_idx: int, delta: float = 1.0):
        """Apply user feedback to a chunk by index. Positive delta => boost, negative => penalize."""
        if not hasattr(self, 'feedback_boosts'):
            self.feedback_boosts = defaultdict(float)
        self.feedback_boosts[chunk_idx] += float(delta)

    def get_feedback(self, chunk_idx: int) -> float:
        if not hasattr(self, 'feedback_boosts'):
            self.feedback_boosts = defaultdict(float)
        return float(self.feedback_boosts.get(chunk_idx, 0.0))

    def search_similar_chunks(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """搜索相似的文本块"""
        from utils.embedding_utils import create_embeddings

        query_embedding = create_embeddings(query)[0]
        
        similarities = []
        # retrieve feedback boosts if present
        feedbacks = getattr(self, 'feedback_boosts', {})

        for i, emb in enumerate(self.text_embeddings):
            base_similarity = cosine_similarity([query_embedding], [emb])[0][0]
            boost = float(feedbacks.get(i, 0.0))
            # weight alpha will be applied by caller or via config; default to 0.1 here if not provided
            from config import config_manager
            try:
                alpha = config_manager.get_config().get('feedback_alpha', 0.1)
            except Exception:
                alpha = 0.1
            final_score = base_similarity + alpha * boost
            similarities.append((i, final_score))

        sorted_indices = sorted(similarities, key=lambda x: x[1], reverse=True)
        return [(self.text_chunks[i], score) for i, score in sorted_indices[:k]]

    def search_similar_chunks_with_index(self, query: str, k: int = 3) -> List[Tuple[int, str, float]]:
        """Return list of (index, chunk_text, score) for top-k similar chunks (uses feedback-adjusted score)."""
        from utils.embedding_utils import create_embeddings

        query_embedding = create_embeddings(query)[0]
        results = []
        feedbacks = getattr(self, 'feedback_boosts', {})
        from config import config_manager
        try:
            alpha = config_manager.get_config().get('feedback_alpha', 0.1)
        except Exception:
            alpha = 0.1

        for i, emb in enumerate(self.text_embeddings):
            base_similarity = cosine_similarity([query_embedding], [emb])[0][0]
            boost = float(feedbacks.get(i, 0.0))
            final_score = base_similarity + alpha * boost
            results.append((i, self.text_chunks[i], final_score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]

    def hybrid_search(self, query: str, k: int = 3) -> List[str]:
        """混合搜索"""
        chunk_results = self.search_similar_chunks(query, k * 2)
        return [chunk for chunk, _ in chunk_results[:k]]

    def get_stats(self) -> Dict:
        """获取向量存储统计信息"""
        return {
            "total_chunks": len(self.text_chunks),
            "total_questions": len(self.question_embeddings),
            "sources": list(set(meta["source_file"] for meta in self.metadata.values() if meta["source_file"]))
        }

    def add_chat_history(self, question: str, answer: str):
        """添加聊天历史"""
        history_item = {
            "id": len(self.chat_history) + 1,
            "question": question,
            "answer": answer,
            "timestamp": str(np.datetime64('now'))
        }
        self.chat_history.append(history_item)

    def get_chat_history(self) -> List[Dict]:
        """获取聊天历史"""
        return self.chat_history

    def clear_chat_history(self):
        """清空聊天历史"""
        self.chat_history.clear()

    # ---------------- Persistence ----------------
    def to_dict(self) -> dict:
        """转换为可序列化的字典形式"""
        return {
            "text_chunks": self.text_chunks,
            "text_embeddings": self.text_embeddings,
            "question_embeddings": self.question_embeddings,
            "chunk_to_questions": {str(k): v for k, v in self.chunk_to_questions.items()},
            "question_to_chunk": {str(k): v for k, v in self.question_to_chunk.items()},
            "chunk_md5_map": {str(k): v for k, v in self.chunk_md5_map.items()},
            "metadata": {str(k): v for k, v in self.metadata.items()},
            "chat_history": self.chat_history,
            "feedback_boosts": {str(k): float(v) for k, v in getattr(self, 'feedback_boosts', {}).items()}
        }

    @classmethod
    def from_dict(cls, data: dict):
        """从字典恢复 VectorStore 实例"""
        vs = cls()
        vs.text_chunks = data.get("text_chunks", [])
        vs.text_embeddings = data.get("text_embeddings", [])
        vs.question_embeddings = data.get("question_embeddings", [])
        vs.chunk_to_questions = defaultdict(list, {int(k): v for k, v in data.get("chunk_to_questions", {}).items()})
        vs.question_to_chunk = {int(k): v for k, v in data.get("question_to_chunk", {}).items()}
        vs.chunk_md5_map = {int(k): v for k, v in data.get("chunk_md5_map", {}).items()}
        vs.metadata = {int(k): v for k, v in data.get("metadata", {}).items()}
        vs.chat_history = data.get("chat_history", [])
        fb = data.get("feedback_boosts", {})
        vs.feedback_boosts = defaultdict(float, {int(k): float(v) for k, v in fb.items()})
        return vs

    def save_to_file(self, path: str) -> bool:
        """将向量存储保存为 JSON 文件（只保存元数据和向量列表）。"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存向量存储失败：{e}")
            return False

    @classmethod
    def load_from_file(cls, path: str):
        """从 JSON 文件加载向量存储实例。"""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            raise RuntimeError(f"加载向量存储失败：{e}")

