"""数据库轮询触发源（README §4.1）。

SQLAlchemy Core 抽象，SQL Server 主力（pyodbc），同一套代码兼容 MySQL/PG/Oracle/SQLite。
每 N 秒：SELECT status=0 → UPDATE 抢占置 1（防多实例重复消费）→ 入队；
执行完把 2(成功)/3(失败) 与 err_msg 写回。
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from functools import partial

logger = logging.getLogger(__name__)


class DbPoller(threading.Thread):
    def __init__(self, settings, engine):
        super().__init__(name="db-poller", daemon=True)
        self.s = settings
        self.engine = engine          # 执行引擎（不是 SQLAlchemy engine）
        self._stop = threading.Event()
        self._db = None
        self._table = None

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------

    def _connect(self):
        import sqlalchemy as sa

        if self._db is None:
            self._db = sa.create_engine(self.s.db.url, pool_pre_ping=True)
            self._table = sa.Table(
                self.s.db.table, sa.MetaData(),
                sa.Column("id", sa.BigInteger, primary_key=True),
                sa.Column("vehicle_no", sa.String(32)),
                sa.Column("status", sa.SmallInteger),
                sa.Column("retry_count", sa.Integer),
                sa.Column("err_msg", sa.String(255)),
                sa.Column("created_at", sa.DateTime),
                sa.Column("updated_at", sa.DateTime),
            )
        return self._db, self._table

    def run(self) -> None:
        if not self.s.db.enabled or not self.s.db.url:
            logger.info("DB 轮询未启用")
            return
        logger.info("DB 轮询启动：每 %.1fs，表 %s", self.s.db.poll_interval, self.s.db.table)
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as e:
                logger.error("DB 轮询出错: %s（10s 后重试）", e)
                self._stop.wait(10.0)
            self._stop.wait(self.s.db.poll_interval)

    def _poll_once(self) -> None:
        import sqlalchemy as sa

        from ..engine import Task

        db, t = self._connect()
        with db.connect() as conn:
            rows = conn.execute(
                sa.select(t.c.id, t.c.vehicle_no)
                .where(t.c.status == 0)
                .order_by(t.c.id)
                .limit(self.s.db.batch)          # SQL Server 方言自动生成 TOP
            ).all()

        for row_id, vehicle_no in rows:
            with db.begin() as conn:
                claimed = conn.execute(
                    sa.update(t)
                    .where(t.c.id == row_id, t.c.status == 0)   # 抢占，防多实例重复消费
                    .values(status=1, updated_at=datetime.now())
                ).rowcount
            if claimed:
                self.engine.submit(Task(
                    vehicle_no=str(vehicle_no).strip(),
                    source="db",
                    db_id=int(row_id),
                    on_done=partial(self._writeback, int(row_id)),
                ))

    def _writeback(self, row_id: int, result) -> None:
        import sqlalchemy as sa

        db, t = self._connect()
        with db.begin() as conn:
            conn.execute(
                sa.update(t)
                .where(t.c.id == row_id)
                .values(
                    status=2 if result.ok else 3,
                    err_msg=None if result.ok else (result.message or "")[:250],
                    updated_at=datetime.now(),
                )
            )
        logger.info("任务 #%d 已回写 status=%d", row_id, 2 if result.ok else 3)
