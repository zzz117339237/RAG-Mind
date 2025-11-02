from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import asyncio
from dotenv import load_dotenv
import logging

load_dotenv()

# 基本日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("RAG-Mind")

from config import config_manager

from models.vector_store import VectorStore
from models.rag_engine import RAGEngine

app = FastAPI(title="RAG-Mind 问答系统", description="公司规章制度智能问答系统")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


config = config_manager.get_config()
vector_store = VectorStore()

rag_engine = RAGEngine(vector_store)


@app.on_event("startup")
async def startup_event():
    """应用启动时调度公司数据处理任务到后台，避免阻塞 FastAPI 启动流程。"""
    logger.info("启动：配置检查并调度公司数据后台处理任务（如需要）...")

    config = config_manager.get_config()
    vector_store_path = config.get("vector_store_path")

    async def _run_process():
        try:
            logger.info("后台任务：开始处理数据")
            # 在线程池中运行同步的处理函数以避免阻塞事件循环
            result = await asyncio.to_thread(rag_engine.process_company_data)
            if isinstance(result, dict) and result.get("success"):
                logger.info("后台任务：处理完成：%s", result.get("message"))
            else:
                logger.error("后台任务：处理返回非预期结果：%s", str(result))
        except Exception as e:
            logger.exception("后台任务：处理数据时发生异常: %s", e)

    # 如果配置开启了持久化并且文件存在，优先加载以避免重复处理
    try:
        if config.get("persist_vector_store") and vector_store_path and os.path.exists(vector_store_path):
            logger.info(f"检测到持久化向量存储 {vector_store_path}，已加载并跳过重建")
            return
    except Exception:
        # 任意检查失败时继续创建后台任务以确保索引可用
        logger.warning("检查持久化向量存储时发生异常，将触发后台处理")

    # 创建后台任务并保存到 app.state 以避免被垃圾回收
    task = asyncio.create_task(_run_process())
    app.state.rebuild_task = task
    logger.info("后台任务已触发，可通过 /api/admin/rebuild_status 查询状态")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/query")
async def query_rag(question_data: dict):
    question = question_data.get("question", "")
    if not question:
        return JSONResponse({"success": False, "message": "问题不能为空"})

    result = rag_engine.query(question)
    return JSONResponse({"success": True, "data": result})


@app.get("/api/history")
async def get_history():
    history = vector_store.get_chat_history()
    return JSONResponse({"success": True, "data": history})


@app.delete("/api/history/clear")
async def clear_history():
    vector_store.clear_chat_history()
    return JSONResponse({"success": True, "message": "聊天历史已清空"})


@app.get("/api/status")
async def get_status():
    stats = vector_store.get_stats()
    return JSONResponse({"success": True, "data": stats})


@app.post("/api/admin/rebuild_index")
async def admin_rebuild_index():
    """管理端点：在后台异步触发索引/向量存储的重建。

    返回说明任务是否已被触发或已有任务运行中。
    """
    # 如果已有任务在运行，则拒绝重复触发
    existing = getattr(app.state, "rebuild_task", None)
    if existing and not existing.done():
        return JSONResponse({"success": False, "message": "已有重建任务正在运行"})

    async def _run_rebuild():
        try:
            logger.info("管理操作：开始异步重建索引（在线程池中执行）")
            result = await asyncio.to_thread(rag_engine.process_company_data)
            if isinstance(result, dict) and result.get("success"):
                logger.info("管理操作：索引重建完成：%s", result.get("message"))
            else:
                logger.error("管理操作：索引重建失败或返回非预期结果: %s", str(result))
        except Exception as e:
            logger.exception("管理操作：索引重建发生异常: %s", e)

    task = asyncio.create_task(_run_rebuild())
    app.state.rebuild_task = task
    return JSONResponse({"success": True, "message": "索引重建任务已在后台触发"})


@app.get("/api/admin/rebuild_status")
async def admin_rebuild_status():
    task = getattr(app.state, "rebuild_task", None)
    if not task:
        return JSONResponse({"success": True, "status": "idle", "message": "未触发过重建任务"})
    return JSONResponse({
        "success": True,
        "status": "running" if not task.done() else "finished",
        "done": task.done()
    })


@app.post("/api/feedback")
async def submit_feedback(payload: dict):
    """接收前端提交的反馈，payload 示例：{ "chunk_indices": [1,2], "action": "up" }

    action: up/down 或 delta numeric
    """
    try:
        chunk_indices = payload.get("chunk_indices") or payload.get("indices")
        if not chunk_indices:
            return JSONResponse({"success": False, "message": "缺少 chunk_indices"})

        action = payload.get("action", "up")
        if action == "up":
            delta = 1.0
        elif action == "down":
            delta = -1.0
        else:
            try:
                delta = float(action)
            except Exception:
                delta = 1.0

        # Apply feedback to each chunk
        for idx in chunk_indices:
            try:
                vector_store.apply_feedback(int(idx), delta=float(delta))
            except Exception:
                logger.exception("应用反馈时发生错误: %s", idx)

        # 可选：立即持久化
        cfg = config_manager.get_config()
        if cfg.get("persist_vector_store") and cfg.get("feedback_persist_on_apply"):
            try:
                vector_store.save_to_file(cfg.get("vector_store_path"))
            except Exception:
                logger.exception("保存向量存储时发生错误")

        return JSONResponse({"success": True, "message": "反馈已提交"})
    except Exception as e:
        logger.exception("提交反馈失败: %s", e)
        return JSONResponse({"success": False, "message": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
