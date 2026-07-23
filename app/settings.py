"""config.yaml 读写 + keyring 凭据。

README §6：运行参数持久化到 config.yaml；元素定位在 app/locators.py（代码维护）；
登录凭据经 keyring 存 Windows 凭据管理器（DPAPI 加密），不落明文。
"""
from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

try:
    import keyring
except ImportError:  # 允许无 keyring 的开发环境导入本模块
    keyring = None

logger = logging.getLogger(__name__)

# 打包后（PyInstaller，sys.frozen）运行根目录：
#   Windows → exe 所在目录（config.yaml / logs / templates / flows.yaml 落 exe 旁，可编辑）
#   macOS   → ~/Library/Application Support/MachineVisionWorkflowAssistant（.app 包内不宜写文件）
# 源码运行 → 仓库根目录。
if getattr(sys, "frozen", False):
    if sys.platform == "darwin":
        ROOT_DIR = Path.home() / "Library" / "Application Support" / "MachineVisionWorkflowAssistant"
    else:
        ROOT_DIR = Path(sys.executable).resolve().parent
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
TEMPLATES_DIR = ROOT_DIR / "templates"
LOGS_DIR = ROOT_DIR / "logs"
SNAPSHOTS_DIR = LOGS_DIR / "snapshots"
DB_SEEN_PATH = ROOT_DIR / "db_seen.sqlite"   # 只读模式的本机去重库（不碰客户系统）
KEYRING_SERVICE = "AutoPrintDeliveryOrder"


@dataclass
class EngineCfg:
    enabled: bool = True                 # 启动后是否立即运行执行引擎
    auto_login: bool = False             # 自动登录总开关（README §5，默认关）
    task_retries: int = 1                # 单任务失败重试次数
    max_consecutive_failures: int = 3    # 连续失败熔断阈值，连败即暂停引擎
    step_timeout: float = 10.0           # 每步等待元素出现的超时（秒）
    dispatch_timeout: float = 120.0      # 反应式派发单任务总超时（无进展多久判失败）
    state_retries: int = 3               # 页面状态识别重试次数（保留，供手动诊断/兼容）
    refresh_wait: float = 3.0            # F5 刷新后等待页面加载秒数
    verify_input: bool = True            # 流程④：注入后 OCR 回读校验
    verify_success: bool = False         # 流程⑥：打印成功特征校验（页面无可靠反馈保持 false）
    monitor: int = 1                     # 目标显示器编号（mss，1=主屏）
    window_title: str = ""               # 浏览器窗口标题包含串；非空则执行前激活并最大化


@dataclass
class DbCfg:
    """数据库轮询触发。装机时直接写 SQL 适配任意库表/视图/存储过程，程序不假设 schema。

    数据库类型由连接串前缀决定（db_type 仅作 UI 填充连接串/SQL 模板的助手）：
    SQL Server=mssql+pyodbc://  MySQL=mysql+pymysql://  PG=postgresql+psycopg://
    Oracle=oracle+oracledb://  SQLite=sqlite:///。

    默认**只读模式**（writeback=False）：对客户库只执行 sql_poll 一条 SELECT，客户只需给只读账号；
    「已处理」状态记在本机 db_seen.sqlite（主键去重），完全不改动对方业务表 —— 不入侵别人系统。
    仅当客户明确授权回写时才开 writeback：每轮对每行执行 sql_claim 抢占（rowcount>0 才算抢到，
    防多实例重复消费）→ 入队执行 → 完成后按结果执行 sql_success / sql_fail 回写。
    两种模式下 sql_poll 的**前两列须为 主键、车辆自编号**。
    绑定参数（自动注入、防注入）：:id=主键，:vehicle_no=车号，:err=失败原因。
    """

    enabled: bool = False
    db_type: str = "mssql"               # 连接串/SQL 模板填充助手；真实驱动以连接串为准
    url: str = "mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17+for+SQL+Server"
    writeback: bool = False              # 是否回写对方库（默认只读，本机 db_seen.sqlite 去重）
    poll_interval: float = 3.0
    batch: int = 5                       # 每轮最多处理条数（Python 侧截断，规避方言差异）
    flow: str = "print_order"            # 该来源的任务走哪个流程（流程编排页 key）
    sql_poll: str = (
        "SELECT id, vehicle_no FROM print_task WHERE status = 0 ORDER BY id")
    sql_claim: str = (
        "UPDATE print_task SET status = 1, updated_at = GETDATE() "
        "WHERE id = :id AND status = 0")
    sql_success: str = (
        "UPDATE print_task SET status = 2, updated_at = GETDATE() WHERE id = :id")
    sql_fail: str = (
        "UPDATE print_task SET status = 3, err_msg = :err, updated_at = GETDATE() "
        "WHERE id = :id")


@dataclass
class ApiCfg:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 18080
    token: str = ""                      # 非空则要求 Authorization: Bearer <token>


@dataclass
class LogCfg:
    level: str = "INFO"


def _from_dict(cls, d):
    """从 dict 构造 dataclass，忽略未知键（配置文件向后兼容）。"""
    if not isinstance(d, dict):
        d = {}
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Settings:
    engine: EngineCfg = field(default_factory=EngineCfg)
    db: DbCfg = field(default_factory=DbCfg)
    api: ApiCfg = field(default_factory=ApiCfg)
    log: LogCfg = field(default_factory=LogCfg)
    path: Path = CONFIG_PATH

    @classmethod
    def load(cls, path: str | Path = CONFIG_PATH) -> "Settings":
        path = Path(path)
        raw = {}
        if path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        s = cls(
            engine=_from_dict(EngineCfg, raw.get("engine")),
            db=_from_dict(DbCfg, raw.get("db")),
            api=_from_dict(ApiCfg, raw.get("api")),
            log=_from_dict(LogCfg, raw.get("log")),
            path=path,
        )
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            s.save()
        return s

    def save(self) -> None:
        data = {
            "engine": asdict(self.engine),
            "db": asdict(self.db),
            "api": asdict(self.api),
            "log": asdict(self.log),
        }
        self.path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        logger.info("配置已保存到 %s", self.path)

    # ---- 登录凭据（keyring → Windows 凭据管理器，DPAPI 加密）----

    def get_credentials(self) -> tuple[str, str] | None:
        if keyring is None:
            raise RuntimeError("keyring 未安装，无法读取登录凭据（pip install keyring）")
        user = keyring.get_password(KEYRING_SERVICE, "username")
        pwd = keyring.get_password(KEYRING_SERVICE, "password")
        if user and pwd:
            return user, pwd
        return None

    def set_credentials(self, username: str, password: str) -> None:
        if keyring is None:
            raise RuntimeError("keyring 未安装，无法保存登录凭据（pip install keyring）")
        keyring.set_password(KEYRING_SERVICE, "username", username)
        keyring.set_password(KEYRING_SERVICE, "password", password)
        logger.info("登录凭据已写入系统凭据管理器（用户: %s）", username)
