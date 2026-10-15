"""结构化运行日志：root logger 双输出（控制台可读 + JSONL 文件），供复盘工具消费。

用法：main.py 启动时 `setup_logging(cfg["log_dir"])`，返回本次运行的 jsonl 文件路径。
业务侧照常 `logging.getLogger("fsm").info(..., extra={"fsm_state": ..., "t": ...})`，
extra 字段会被 JSONL formatter 原样序列化（保留键除外）。
"""
import json
import logging
import time
from pathlib import Path

# LogRecord 的保留属性：这些不算业务 extra 字段
_RESERVED = set(logging.LogRecord(
    "", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}

_RUN_MARKER = "_gongchuang_run_handler"


class JsonlFormatter(logging.Formatter):
    """每行一个 JSON 对象：{t, level, logger, msg, **extra}（extra 可覆盖 t 等基准键）。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "t": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items()
                        if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_dir: str) -> Path:
    """初始化 root logger：控制台可读 + JSONL 文件双输出，返回本次 jsonl 路径。

    重复调用（如多个测试各调一次）会先移除上一次由本函数安装的处理程序，
    避免 handler 堆积；未由本函数安装的用户 handler 不受影响。
    """
    dir_path = Path(log_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000:03d}.jsonl"

    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _RUN_MARKER, False):
            root.removeHandler(h)
            h.close()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    setattr(console, _RUN_MARKER, True)
    root.addHandler(console)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(JsonlFormatter())
    setattr(file_handler, _RUN_MARKER, True)
    root.addHandler(file_handler)

    root.setLevel(logging.INFO)
    return path
