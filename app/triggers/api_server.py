"""HTTP API 触发源（README §4.2）：FastAPI + uvicorn，默认 127.0.0.1:18080。

注意：本文件不可用 `from __future__ import annotations`——PrintRequest 定义在
create_app 内部，字符串化注解会让 FastAPI 解析不到该类型（被误判为 query 参数）。
"""
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def create_app(engine, settings):
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field

    from ..engine import Task

    class PrintRequest(BaseModel):
        vehicle_no: str = Field(min_length=1, max_length=64)
        task_id: Optional[str] = None          # 外部单号（可选）
        callback_url: Optional[str] = None     # 结果回调（可选）

    app = FastAPI(title="AutoPrintDeliveryOrder", version="0.1.0")

    def _check_auth(authorization: str | None) -> None:
        token = settings.api.token
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid token")

    @app.post("/print", status_code=202)
    def submit_print(req: PrintRequest, authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        vehicle_no = req.vehicle_no.strip()
        if not vehicle_no:
            raise HTTPException(status_code=422, detail="vehicle_no 不能为空")
        size = engine.submit(Task(
            vehicle_no=vehicle_no, source="api",
            external_id=req.task_id, callback_url=req.callback_url,
        ))
        return {"queued": True, "queue_size": size}

    @app.get("/status")
    def status(authorization: Optional[str] = Header(default=None)):
        _check_auth(authorization)
        return engine.status()

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


class ApiServer:
    def __init__(self, settings, engine):
        self.s = settings
        self.engine = engine
        self._server = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import uvicorn

        app = create_app(self.engine, self.s)
        config = uvicorn.Config(app, host=self.s.api.host, port=self.s.api.port,
                                log_level="warning")
        self._server = uvicorn.Server(config)
        # 非主线程运行，不装信号处理器
        self._server.install_signal_handlers = lambda: None
        self._thread = threading.Thread(target=self._server.run, name="api", daemon=True)
        self._thread.start()
        logger.info("API 服务已启动: http://%s:%d", self.s.api.host, self.s.api.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
