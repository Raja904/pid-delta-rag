"""
observability/tracer.py
Lightweight request tracer. Writes a JSON trace file per request to ./traces/.
Each trace captures: stage timings, LLM calls (prompt/response/tokens/cost),
and errors. No external dependency -- pure stdlib.
"""
from __future__ import annotations
import json, time, uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

TRACES_DIR = Path(__file__).parent.parent.parent / "traces"
TRACES_DIR.mkdir(exist_ok=True)

# Groq pricing (approx, USD per 1M tokens) -- update if model changes
_COST_PER_1M = {
    "llama3-70b-8192":   {"input": 0.59, "output": 0.79},
    "llama3-8b-8192":    {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768":{"input": 0.24, "output": 0.24},
    "default":           {"input": 0.59, "output": 0.79},
}


@dataclass
class LLMCall:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    prompt_preview: str   # first 300 chars
    response_preview: str # first 300 chars
    error: str = ""


@dataclass
class Stage:
    name: str
    start: float = field(default_factory=time.time)
    end: float   = 0.0
    error: str   = ""

    @property
    def latency_s(self) -> float:
        return round(self.end - self.start, 4) if self.end else 0.0


@dataclass
class Trace:
    request_id: str
    operation:  str
    start:      float = field(default_factory=time.time)
    end:        float = 0.0
    stages:     list[Stage]    = field(default_factory=list)
    llm_calls:  list[LLMCall] = field(default_factory=list)
    metadata:   dict           = field(default_factory=dict)
    error:      str            = ""

    @property
    def total_latency_s(self) -> float:
        return round(self.end - self.start, 4) if self.end else 0.0

    @property
    def total_cost_usd(self) -> float:
        return round(sum(c.cost_usd for c in self.llm_calls), 6)

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.llm_calls)


# Module-level current trace (simple, not async-safe -- fine for CLI/Streamlit)
_current_trace: Trace | None = None


def start_trace(operation: str, metadata: dict | None = None) -> Trace:
    global _current_trace
    rid = str(uuid.uuid4())[:8]
    _current_trace = Trace(
        request_id = rid,
        operation  = operation,
        metadata   = metadata or {},
    )
    return _current_trace


def get_trace() -> Trace | None:
    return _current_trace


@contextmanager
def trace_stage(name: str):
    """Context manager for timing a named pipeline stage."""
    stage = Stage(name=name, start=time.time())
    if _current_trace:
        _current_trace.stages.append(stage)
    try:
        yield stage
    except Exception as e:
        stage.error = str(e)
        raise
    finally:
        stage.end = time.time()


def record_llm_call(
    model: str,
    prompt: str,
    response: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_s: float,
    error: str = "",
) -> LLMCall:
    pricing = _COST_PER_1M.get(model, _COST_PER_1M["default"])
    cost = (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000
    call = LLMCall(
        model              = model,
        prompt_tokens      = prompt_tokens,
        completion_tokens  = completion_tokens,
        cost_usd           = round(cost, 8),
        latency_s          = round(latency_s, 4),
        prompt_preview     = prompt[:300],
        response_preview   = response[:300],
        error              = error,
    )
    if _current_trace:
        _current_trace.llm_calls.append(call)
    return call


def finish_trace(error: str = "") -> Path | None:
    global _current_trace
    if not _current_trace:
        return None
    t = _current_trace
    t.end   = time.time()
    t.error = error

    out_path = TRACES_DIR / f"{t.request_id}_{t.operation}.json"
    with open(out_path, "w") as f:
        json.dump(_trace_to_dict(t), f, indent=2)

    _current_trace = None
    return out_path


def _trace_to_dict(t: Trace) -> dict:
    return {
        "request_id":     t.request_id,
        "operation":      t.operation,
        "total_latency_s":t.total_latency_s,
        "total_cost_usd": t.total_cost_usd,
        "total_tokens":   t.total_tokens,
        "error":          t.error,
        "metadata":       t.metadata,
        "stages": [
            {"name": s.name, "latency_s": s.latency_s, "error": s.error}
            for s in t.stages
        ],
        "llm_calls": [
            {
                "model":            c.model,
                "prompt_tokens":    c.prompt_tokens,
                "completion_tokens":c.completion_tokens,
                "cost_usd":         c.cost_usd,
                "latency_s":        c.latency_s,
                "prompt_preview":   c.prompt_preview,
                "response_preview": c.response_preview,
                "error":            c.error,
            }
            for c in t.llm_calls
        ],
    }
