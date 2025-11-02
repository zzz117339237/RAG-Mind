from typing import List, Dict, Any
import logging
from sklearn.metrics.pairwise import cosine_similarity
from utils.embedding_utils import create_embeddings, client

logger = logging.getLogger(__name__)


class QueryRewriter:
    @staticmethod
    def rewrite_query(original_query: str, model_name: str = "qwen-max") -> str:
        prompt = (
            f"请重写以下查询，使其更适合在文档中搜索相关信息。保持原意不变，但可以使表述更清晰、更具体。\n"
            f"原始查询: {original_query}\n"
            "只返回改写后的查询，不要添加任何解释。"
        )
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "你是一个查询改写专家，能优化查询表述使其更适合信息检索。"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"查询改写失败，返回原查询：{e}")
            return original_query

    @staticmethod
    def generate_subqueries(original_query: str, num_subqueries: int = 3, model_name: str = "qwen-max") -> List[str]:
        prompt = (
            f"请将以下查询分解为{num_subqueries}个更简单的子查询，这些子查询的答案组合起来应该能够回答原始查询。\n"
            f"原始查询: {original_query}\n"
            "请以列表形式返回子查询，每个子查询占一行，不要添加额外说明。"
        )
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": "你是一个查询分解专家，能将复杂查询分解为简单的子查询。"},
                    {"role": "user", "content": prompt}
                ]
            )
            subqueries = response.choices[0].message.content.strip().split("\n")
            subqueries = [q.split('. ', 1)[-1].strip() for q in subqueries if q.strip()]
            return subqueries[:num_subqueries]
        except Exception as e:
            logger.warning(f"生成子查询失败：{e}")
            # fallback: simple heuristic split
            parts = original_query.replace('，', ' ').replace(',', ' ').split()
            return [' '.join(parts[i::num_subqueries]) for i in range(num_subqueries)]


def semantic_search(query: str, text_chunks: List[str], embeddings: List[List[float]] = None, k: int = 3) -> List[str]:
    if embeddings is None:
        embeddings = create_embeddings(text_chunks)
    query_embedding = create_embeddings(query)[0]
    scores = [cosine_similarity([query_embedding], [emb])[0][0] for emb in embeddings]
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [text_chunks[i] for i in idxs]


def context_enriched_search(search_query: str, chunked_texts: List[str], chunk_embeddings: List[List[float]],
                            top_k: int = 1, context_window_size: int = 1) -> List[str]:
    query_embedding = create_embeddings(search_query)[0]
    similarity_list = []
    for idx, chunk_embedding in enumerate(chunk_embeddings):
        similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
        similarity_list.append((idx, similarity))
    similarity_list.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarity_list[:top_k]]
    result_indices = set()
    for idx in top_indices:
        start_idx = max(0, idx - context_window_size)
        end_idx = min(len(chunked_texts) - 1, idx + context_window_size)
        for context_idx in range(start_idx, end_idx + 1):
            result_indices.add(context_idx)
    return [chunked_texts[i] for i in sorted(result_indices)]


def keyword_rerank(query: str, chunks: List[str], k: int = 3, weight: float = 0.6) -> List[str]:
    stop_words = {"的", "了", "是", "在", "有", "和", "及", "与", "等", "能", "可", "为", "对", "将"}
    query_words = [w.strip() for w in query.replace('?', ' ').replace('。', ' ').split() if w.strip() and w not in stop_words]
    if not query_words:
        return chunks[:k]
    chunk_keyword_scores = []
    for chunk in chunks:
        chunk_words = [w.strip() for w in chunk.replace('?', ' ').replace('。', ' ').split() if w.strip() and w not in stop_words]
        if not chunk_words:
            chunk_keyword_scores.append(0.0)
            continue
        common_words = set(query_words) & set(chunk_words)
        keyword_score = len(common_words) / len(query_words)
        chunk_keyword_scores.append(keyword_score)
    query_emb = create_embeddings(query)[0]
    chunk_embs = create_embeddings(chunks)
    semantic_scores = [cosine_similarity([query_emb], [emb])[0][0] for emb in chunk_embs]
    fused_scores = [(kws * weight) + (ss * (1 - weight)) for kws, ss in zip(chunk_keyword_scores, semantic_scores)]
    sorted_indices = sorted(range(len(fused_scores)), key=lambda i: fused_scores[i], reverse=True)
    return [chunks[i] for i in sorted_indices[:k]]


def rerank_results_llm(query: str, chunks: List[str], model_name: str = "qwen-max", k: int = 3) -> List[str]:
    if len(chunks) <= k:
        return chunks
    prompt = f"请根据与以下查询的相关性，对提供的文本块进行排序。\n查询: {query}\n文本块列表:\n"
    for i, chunk in enumerate(chunks):
        prompt += f"{i + 1}. {chunk[:200]}...\n"
    prompt += "请只返回排序后的索引（用逗号分隔，如：3,1,2），不要添加任何解释。"
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": "你是一个文本排序专家，能根据查询相关性对文本块进行排序。"},
                {"role": "user", "content": prompt}
            ]
        )
        ranked_indices = [int(idx.strip()) - 1 for idx in response.choices[0].message.content.split(",")]
        valid = []
        seen = set()
        for i in ranked_indices:
            if 0 <= i < len(chunks) and i not in seen:
                seen.add(i)
                valid.append(i)
        for i in range(len(chunks)):
            if i not in seen:
                valid.append(i)
        return [chunks[i] for i in valid[:k]]
    except Exception as e:
        logger.warning(f"LLM重排序失败：{e}，使用原始结果")
        return chunks[:k]


def rerank_by_config(query: str, chunks: List[str], config: Dict[str, Any]) -> List[str]:
    method = config.get("rerank_method", "none")
    topk = config.get("retrieval_top_k", 3)
    if method == "none":
        return chunks[:topk]
    if method == "llm":
        return rerank_results_llm(query, chunks, k=topk)
    if method == "keyword":
        return keyword_rerank(query, chunks, k=topk, weight=config.get("keyword_weight", 0.6))
    raise ValueError(f"无效的重排序方法：{method}")


def retrieve_by_config(query: str, vector_store, config: Dict[str, Any], enhanced_chunk_embeddings=None) -> List[str]:
    method = config.get("retrieval_method", "hybrid")
    top_k = config.get("retrieval_top_k", 3)
    if method == "chunk_similar":
        results = vector_store.search_similar_chunks(query, k=top_k * 2)
        return [chunk for chunk, _ in results]
    if method == "question_similar":
        results = vector_store.search_similar_questions(query, k=top_k * 2)
        return [chunk for chunk, _ in results]
    if method == "hybrid":
        return vector_store.hybrid_search(query, k=top_k * 2)
    if method == "query_enhanced":
        rewritten = QueryRewriter.rewrite_query(query)
        orig = vector_store.hybrid_search(query, k=top_k)
        rew = vector_store.hybrid_search(rewritten, k=top_k)
        combined = []
        seen = set()
        for r in orig + rew:
            if r not in seen:
                seen.add(r)
                combined.append(r)
                if len(combined) >= top_k * 2:
                    break
        return combined
    if method == "subquery_enhanced":
        subqueries = QueryRewriter.generate_subqueries(query, num_subqueries=config.get("num_subqueries", 3))
        all_results = []
        seen = set()
        orig = vector_store.hybrid_search(query, k=top_k)
        for r in orig:
            if r not in seen:
                seen.add(r)
                all_results.append(r)
        for subq in subqueries:
            subr = vector_store.hybrid_search(subq, k=top_k)
            for r in subr:
                if r not in seen:
                    seen.add(r)
                    all_results.append(r)
                    if len(all_results) >= top_k * 2:
                        break
            if len(all_results) >= top_k * 2:
                break
        return all_results[:top_k * 2]
    if method == "context_enriched":
        chunk_embeddings = create_embeddings(vector_store.text_chunks)
        return context_enriched_search(query, vector_store.text_chunks, chunk_embeddings,
                                       top_k=top_k, context_window_size=config.get("context_window_size", 1))
    if method == "enhanced_semantic":
        if not enhanced_chunk_embeddings:
            raise ValueError("增强语义检索需要带标题的chunk_embeddings，请确保分块方法为with_header并传入enhanced_chunk_embeddings")
        # enhanced_chunk_embeddings expected as list of dicts with 'text' key
        return [chunk["text"] for chunk in enhanced_chunk_embeddings[:top_k * 2]]
    raise ValueError(f"无效的检索方法：{method}")
