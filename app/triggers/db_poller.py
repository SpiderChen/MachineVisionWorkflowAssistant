"""数据库轮询触发源（README §4.1）。

SQLAlchemy Core，SQL Server 主力（pyodbc），同一套代码兼容 MySQL/PG/Oracle/SQLite。
装机时直接写 SQL（settings.DbCfg），程序不假设任何 schema。

**默认只读模式**（db.writeback=False，推荐）：对客户库只执行 sql_poll 一条 SELECT，
客户只需给只读账号，完全不改动对方业务表 —— 不入侵别人系统。「这行处理过没有」记在
本机 db_seen.sqlite（按主键去重），程序自己兜底，不依赖对方库的状态列。

**回写模式**（db.writeback=True，需客户授权写权限）：每 N 秒 sql_poll 捞取待处理
（前两列=主键/车号）→ sql_claim 抢占（rowcount>0 防多实例重复消费）→ 入队；
执行完按结果跑 sql_success / sql_fail 回写。
绑定参数：:id 主键、:vehicle_no 车号、:err 失败原因（参数化，防注入）。
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from functools import partial

from ..settings import DB_SEEN_PATH

logger = logging.getLogger(__name__)

_SEEN_TTL = 30 * 24 * 3600.0    # 本机去重记录保留 30 天后清理，避免无限增长


class _SeenStore:
    """只读模式下的本机去重库：记哪些主键已提交过，防重复打印。

    单线程（db-poller 线程）访问，连接就地创建。落在 exe/数据目录旁，随程序生命周期持久化。
    """

    def __init__(self, path=DB_SEEN_PATH):
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS seen (pk TEXT PRIMARY KEY, ts REAL)")
        self._conn.commit()
        self._prune()

    def is_seen(self, pk: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM seen WHERE pk = ?", (pk,))
        return cur.fetchone() is not None

    def mark(self, pk: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO seen (pk, ts) VALUES (?, ?)", (pk, time.time()))
        self._conn.commit()

    def _prune(self) -> None:
        self._conn.execute("DELETE FROM seen WHERE ts < ?", (time.time() - _SEEN_TTL,))
        self._conn.commit()


class DbPoller(threading.Thread):
    def __init__(self, settings, engine):
        super().__init__(name="db-poller", daemon=True)
        self.s = settings
        self.engine = engine          # 执行引擎（不是 SQLAlchemy engine）
        self._stop = threading.Event()
        self._db = None
        self._seen = None             # 只读模式的本机去重库，懒加载

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------

    def _connect(self):
        import sqlalchemy as sa

        if self._db is None:
            self._db = sa.create_engine(self.s.db.url, pool_pre_ping=True)
        return self._db

    def _seen_store(self) -> _SeenStore:
        if self._seen is None:
            self._seen = _SeenStore()
        return self._seen

    def run(self) -> None:
        if not self.s.db.enabled or not self.s.db.url:
            logger.info("DB 轮询未启用")
            return
        mode = "回写" if self.s.db.writeback else "只读（本机去重，不改对方库）"
        logger.info("DB 轮询启动：每 %.1fs，%s 模式", self.s.db.poll_interval, mode)
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

        c = self.s.db
        db = self._connect()
        with db.connect() as conn:
            rows = conn.execute(sa.text(c.sql_poll)).all()

        for row in rows[:c.batch]:
            row_id, vehicle_no = row[0], row[1]     # 约定：SELECT 前两列 = 主键、车号
            if not self._claim(row_id):
                continue
            self.engine.submit(Task(
                vehicle_no=str(vehicle_no).strip(),
                source="db",
                flow=c.flow,
                db_id=int(row_id) if str(row_id).lstrip("-").isdigit() else None,
                on_done=partial(self._on_done, row_id),
            ))

    def _claim(self, row_id) -> bool:
        """抢占一行，返回是否可提交。

        回写模式：UPDATE sql_claim，rowcount>0 才算抢到（DB 层互斥，防多实例重复）。
        只读模式：查本机 db_seen —— 没见过就登记为已处理并放行，绝不碰对方库。
        """
        import sqlalchemy as sa

        c = self.s.db
        if c.writeback:
            with self._connect().begin() as conn:
                return bool(conn.execute(sa.text(c.sql_claim), {"id": row_id}).rowcount)

        pk = str(row_id)
        seen = self._seen_store()
        if seen.is_seen(pk):
            return False
        seen.mark(pk)     # 提交前即登记：宁可漏打（可手动补打）也不重复打印
        return True

    def _on_done(self, row_id, result) -> None:
        c = self.s.db
        if not c.writeback:
            logger.info("任务 #%s %s（只读模式，不回写对方库）",
                        row_id, "成功" if result.ok else "失败")
            return
        self._writeback(row_id, result)

    def _writeback(self, row_id, result) -> None:
        import sqlalchemy as sa

        c = self.s.db
        db = self._connect()
        if result.ok:
            sql, params = c.sql_success, {"id": row_id}
        else:
            sql, params = c.sql_fail, {"id": row_id, "err": (result.message or "")[:250]}
        with db.begin() as conn:
            conn.execute(sa.text(sql), params)
        logger.info("任务 #%s 已回写（%s）", row_id, "成功" if result.ok else "失败")
