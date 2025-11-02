import os
from typing import Dict, Any

# 默认配置（集中管理 RAG 相关的方法/参数）
DEFAULT_CONFIG = {
    # ---------------- 分块配置 ----------------
    "chunk_method": "fixed_size",  # 分块方法：fixed_size（固定大小）、semantic（语义分块）、with_header（带标题分块）
    "chunk_size": 512,  # 固定分块/带标题分块的块大小（字符数）
    "overlap_size": 50,  # 固定分块/带标题分块的重叠字符数
    "semantic_chunk_params": {  # 语义分块参数（仅chunk_method="semantic"时生效）
        "method": "percentile",  # 断点计算方法：percentile/standard_deviation/interquartile
        "threshold": 90  # 断点阈值（percentile=90表示90百分位）
    },

    # ---------------- 检索配置 ----------------
    "retrieval_method": "hybrid",  # hybrid/chunk_similar/question_similar/query_enhanced/subquery_enhanced/context_enriched/enhanced_semantic
    "retrieval_top_k": 3,  # 检索返回Top-k结果
    "context_window_size": 1,  # 上下文感知检索的窗口大小（仅retrieval_method="context_enriched"时生效）
    "num_subqueries": 3,  # 子查询增强检索的子查询数量（仅retrieval_method="subquery_enhanced"时生效）

    # ---------------- 重排序配置 ----------------
    "rerank_method": "keyword",  # 重排序方法：none（不重排）、llm（大模型重排）、keyword（关键词排序）
    "keyword_weight": 0.6,  # 关键词排序的权重（与原始相似度的融合比例，0-1）

    # ---------------- 其他配置 ----------------
    "context_processing": "compressed",  # 上下文处理：original（原始）、compressed（压缩）
    "use_feedback": True,  # 是否启用用户反馈优化
    # 用户反馈相关参数
    "feedback_alpha": 0.1,  # 反馈分数对最终排序的影响权重（final_score = base_similarity + alpha * feedback_boost）
    "feedback_persist_on_apply": False,  # 是否在每次提交反馈时立即持久化向量存储

    # 路径与持久化配置
    "data_path": "data",
    "persist_vector_store": True,  # 是否将向量存储持久化到磁盘（JSON）
    "vector_store_path": os.path.join("data", "vector_store.json"),
}


class ConfigManager:
    def __init__(self):
        self.config = DEFAULT_CONFIG.copy()
        self._ensure_directories()

    def _ensure_directories(self):
        """确保必要的目录存在"""
        # data 目录用于存放原始文档
        os.makedirs(self.config["data_path"], exist_ok=True)
        # 静态图片目录（如果前端使用了 /static/img/image.png）
        os.makedirs(os.path.join("static", "img"), exist_ok=True)

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """更新配置"""
        try:
            for key, value in new_config.items():
                if key in self.config:
                    if isinstance(value, dict) and isinstance(self.config[key], dict):
                        self.config[key].update(value)
                    else:
                        self.config[key] = value
            return True
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception(f"配置更新失败: {e}")
            return False


# 全局配置管理器实例
config_manager = ConfigManager()
