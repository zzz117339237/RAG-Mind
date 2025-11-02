# RAG-Mind
雷石-AI参赛项目
一个基于 RAG 思路的公司文档问答服务。提供：

- 后端：FastAPI（入口 `app.py`）
- 文本处理与分块：`utils/text_processor.py`
- 嵌入与向量存储：`utils/embedding_utils.py`、`models/vector_store.py`
# RAG-Mind

RAG-Mind 是一个面向公司/组织内部的文档问答系统，采用 RAG（Retrieval-Augmented Generation）思路：
将公司文档分块、向量化，建立检索能力，并在检索到的上下文上生成回答。

主要功能
- 启动即对 `data/` 目录中的文档进行分块和向量化（后台异步处理，避免阻塞主线程）。
- 使用兼容 OpenAI 的 embeddings 服务为每个文本块生成向量，并在内存中建立向量索引；提供简单的 JSON 持久化接口以便可选保存/加载索引。
- 向量检索以余弦相似度为基础，支持关键词重排序（默认）以及可选的大模型重排序流程。
- 最终回答在检索到的上下文基础上由大模型（chat/completions）生成；当外部模型不可用时，系统提供本地回退逻辑以保证响应。
- 用户反馈机制：前端展示“有帮助/无帮助”按钮，后端接收反馈并将其作为文本块的权重（boost）融入后续检索排序。
- 管理与运维：提供异步重建索引、查询重建状态、查看系统与向量存储统计信息等管理端点。
- 前端：基于 Jinja2 的聊天界面，支持历史记录、清空会话、自动滚动与固定底部的输入区域，优化了移动端与桌面体验。

主要组件
- 后端：FastAPI（入口：`app.py`）
- 文本分块：`utils/text_processor.py`、`utils/file_processor.py`
- 嵌入：`utils/embedding_utils.py`
 - 向量存储/检索：`models/vector_store.py`（内存存储 + 基于余弦相似度的检索；提供简单 JSON 持久化接口 `save_to_file` / `load_from_file`）
- RAG 引擎：`models/rag_engine.py`（文件处理、检索、回答生成）
- 前端：`templates/`（Jinja2）+ `static/`（静态资源），提供简易聊天界面

快速开始
1. 进入项目目录：

```powershell
cd D:\python\python_location\pythonProject\RAG-Mind
```

2. 建议创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. 安装依赖：

```powershell
python -m pip install -r requirements.txt
```


4. 启动服务：

- 开发（带热重载）：

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

- 本地/生产（推荐用于长期运行 / 避免被重载中断）：

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000 --log-level info
```

5. 打开页面： http://127.0.0.1:8000

