import os
import json
import time
import logging
from openai import OpenAI
from typing import List, Union

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

logger = logging.getLogger(__name__)


def create_embeddings(texts: Union[str, List[str]], model: str = "text-embedding-v4") -> List[List[float]]:
    """创建文本的嵌入向量。

    说明：对长度超过服务限制的输入进行分批；对每个分批请求进行重试（默认 3 次）。
    出错时会返回随机向量作为后备，避免阻塞整个流程（开发/测试友好）。
    """
    if isinstance(texts, str):
        texts = [texts]

    try:
        max_batch = 10
        all_embeddings = []
        for i in range(0, len(texts), max_batch):
            batch = texts[i:i + max_batch]
            # 对每个 batch 增加重试
            retries = 3
            backoff = 1
            for attempt in range(1, retries + 1):
                try:
                    completion = client.embeddings.create(
                        model=model,
                        input=batch,
                        encoding_format="float"
                    )
                    data = json.loads(completion.model_dump_json())
                    batch_embs = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(batch_embs)
                    break
                except Exception as e:
                    logger.warning(f"嵌入请求第 {attempt} 次失败: {e}")
                    if attempt < retries:
                        time.sleep(backoff)
                        backoff *= 2
                    else:
                        logger.exception("嵌入请求重试失败，使用随机向量作为回退。")
                        import numpy as np
                        # 为该批次生成与模型相容的随机向量（暂用128维）
                        batch_fallback = [np.random.rand(128).tolist() for _ in batch]
                        all_embeddings.extend(batch_fallback)
        return all_embeddings
    except Exception as e:
        logger.exception(f"嵌入生成失败（总控）: {e}")
        import numpy as np
        return [np.random.rand(128).tolist() for _ in texts]
