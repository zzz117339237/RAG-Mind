from typing import Dict
from models.vector_store import VectorStore
from utils.text_processor import TextProcessor
from utils.semantic_search import retrieve_by_config, rerank_by_config
from utils.embedding_utils import client
from utils.file_processor import FileProcessor
from config import config_manager
import os
import logging

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def process_company_data(self) -> Dict:
        """处理公司数据文件夹中的所有文件并按配置持久化向量存储。"""
        try:
            data_path = config_manager.get_config()["data_path"]
            processed_files = 0
            total_chunks = 0

            if not os.path.exists(data_path):
                return {
                    "success": False,
                    "message": f"数据目录 {data_path} 不存在"
                }

            for filename in os.listdir(data_path):
                file_path = os.path.join(data_path, filename)
                if os.path.isfile(file_path):
                    file_extension = os.path.splitext(filename)[1].lower()
                    if file_extension in FileProcessor.get_supported_extensions():
                        logger.info(f"正在处理文件: {filename}")
                        text, file_name = FileProcessor.extract_text_from_file(file_path)
                        if text.strip():
                            config = config_manager.get_config()
                            chunks = TextProcessor.chunk_by_config(text, config)

                            for chunk in chunks:
                                self.vector_store.add_chunk(chunk, source_file=file_name)

                            processed_files += 1
                            total_chunks += len(chunks)
                            logger.info(f"文件 {filename} 处理完成，生成 {len(chunks)} 个文本块")
                        else:
                            logger.warning(f"文件 {filename} 提取的文本为空")


            return {
                "success": True,
                "message": f"处理完成：共处理 {processed_files} 个文件，生成 {total_chunks} 个文本块",
                "files_processed": processed_files,
                "chunks_created": total_chunks
            }
        except Exception as e:
            logger.exception("处理公司数据时发生异常")
            return {
                "success": False,
                "message": f"处理失败: {str(e)}"
            }

    def query(self, question: str) -> Dict:
        """执行问答查询"""
        config = config_manager.get_config()

        try:
            # 先使用向量存储获取候选（包含索引），然后按配置重排序并返回 top-k
            # 获取候选（index, text, score）
            candidates = self.vector_store.search_similar_chunks_with_index(question, k=config.get('retrieval_top_k', 3) * 2)
            candidate_texts = [t for _, t, _ in candidates]

            # 根据配置进行重排序（返回Top-k的文本）
            topk_texts = rerank_by_config(question, candidate_texts, config)

            # map texts back to indices (first match)
            topk_indices = []
            for t in topk_texts:
                found_idx = None
                for idx, text, score in candidates:
                    if text == t:
                        found_idx = idx
                        break
                topk_indices.append(found_idx if found_idx is not None else -1)

            context = "\n\n".join(topk_texts)
            answer = self._generate_answer(question, context)
            self.vector_store.add_chat_history(question, answer)

            return {
                "question": question,
                "answer": answer,
                "context": context,
                "chunks_used": len(topk_texts),
                "chunk_indices": topk_indices
            }
        except Exception as e:
            error_msg = f"查询处理失败: {str(e)}"
            self.vector_store.add_chat_history(question, error_msg)
            return {
                "question": question,
                "answer": error_msg,
                "context": "",
                "chunks_used": 0
            }

    def _generate_answer(self, question: str, context: str) -> str:
        """使用大模型生成最终回答；若 LLM 调用失败则回退到本地规则。

        将问题和检索到的上下文作为 prompt 发送给 chat/completions，temperature 设为 0
        以获得确定性回答。
        """
        if not context:
            return "抱歉，我没有找到相关信息来回答您的问题。"

        # 构造 prompt，注意截断上下文以避免超长
        max_context_chars = 4000
        trimmed_context = context if len(context) <= max_context_chars else context[-max_context_chars:]

        system_msg = (
            "你是一个能够基于给定上下文回答问题的助手。只使用提供的上下文信息来回答，"
            "不要编造其他信息。如果上下文不足以回答，请说明找不到相关信息。"
        )

        user_msg = f"问题: {question}\n\n上下文:\n{trimmed_context}"

        try:
            response = client.chat.completions.create(
                model="qwen-max",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=512
            )
            # 兼容不同 client 返回结构
            content = None
            try:
                content = response.choices[0].message.content.strip()
            except Exception:
                try:
                    # 有些实现可能序列化为 dict-like
                    content = response.choices[0]["message"]["content"].strip()
                except Exception:
                    pass

            if content:
                return content
            else:
                logger.warning("LLM 未返回有效内容，回退到规则化回答。")
        except Exception as e:
            logger.warning(f"调用 LLM 生成回答失败，回退到本地生成: {e}")

        # 回退：原有的简单规则
        question_lower = question.lower()
        if "什么" in question_lower or "介绍" in question_lower:
            return f"根据公司文档内容，相关信息如下：\n\n{trimmed_context[:800]}..."
        elif "如何" in question_lower or "怎么" in question_lower:
            return f"操作指南：\n\n{trimmed_context[:800]}..."
        elif "为什么" in question_lower:
            return f"原因分析：\n\n{trimmed_context[:800]}..."
        else:
            return f"相关内容：\n\n{trimmed_context[:800]}..."
