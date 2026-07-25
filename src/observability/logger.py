"""
observability/logger.py
Structured JSON logger with correlation/request id support.
Every log line is valid JSON -- no free-text.
"""
from __future__ import annotations
import json, logging, time, uuid
from contextvars import ContextVar
from typing import Any

# Thread/async-safe request ID context variable
_request_id: ContextVar[str] = ContextVar("request_id", default="no-request")

def set_request_id(rid: str) -> None:
    _request_id.set(rid)

def get_request_id() -> str:
    return _request_id.get()

def new_request_id() -> str:
    rid = str(uuid.uuid4())[:8]
    _request_id.set(rid)
    return rid


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level":      record.levelname,
            "request_id": _request_id.get(),
            "module":     record.module,
            "msg":        record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # merge any extra fields passed via extra={...}
        for k, v in record.__dict__.items():
            if k not in ("msg", "args", "levelname", "levelno", "pathname",
                         "filename", "module", "exc_info", "exc_text",
                         "stack_info", "lineno", "funcName", "created",
                         "msecs", "relativeCreated", "thread", "threadName",
                         "processName", "process", "name", "message",
                         "taskName"):
                payload[k] = v
        return json.dumps(payload)


def get_logger(name: str = "delta_chat") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


log = get_logger()
