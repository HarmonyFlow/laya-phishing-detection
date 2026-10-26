"""Laya phishing detection.

An email is scored with a general phishing question. Only when that score stays
below the cascade threshold does a second business-email-compromise question
run; the higher of the two answers is returned. Inference is serialized on the
CPU and runs fresh for every request.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import socket
import sys
import threading
import time
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal


def _missing_dependency() -> str | None:
    for module in ("fastapi", "pydantic", "uvicorn", "torch", "transformers", "laya"):
        if importlib.util.find_spec(module) is None:
            return module
    return None


def _require_dependencies() -> None:
    """Fail with an actionable hint instead of a bare ModuleNotFoundError."""
    missing = _missing_dependency()
    if missing is None:
        return
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    start_hint = "py -3.11 main.py" if os.name == "nt" else "python3.11 main.py"
    raise SystemExit(
        f"Dependencies are missing ({missing}) for Python {version}.\n"
        f"Interpreter: {sys.executable}\n"
        "Install them for this interpreter with:\n"
        "  python -m pip install -r requirements.txt\n"
        "Or start with the interpreter where they are installed, for example:\n"
        f"  {start_hint}"
    )


_require_dependencies()

# Environment and thread settings must be in place before torch is imported.
DEFAULT_THREADS = max(1, min(12, os.cpu_count() or 12))


def _thread_count() -> int:
    try:
        return max(1, int(os.getenv("LAYA_THREADS", DEFAULT_THREADS)))
    except ValueError:
        return DEFAULT_THREADS


THREADS = _thread_count()
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
warnings.filterwarnings(
    "ignore",
    message="laya: this checkpoint ships invalid temperatures.*",
    category=RuntimeWarning,
)

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from laya import Router  # noqa: E402
from pydantic import (  # noqa: E402
    BaseModel,
    Field,
    StringConstraints,
    field_validator,
)
from pydantic_core import PydanticCustomError  # noqa: E402

torch.set_num_threads(THREADS)
with suppress(RuntimeError, ValueError):
    # The inter-op pool can only be configured once per process.
    torch.set_num_interop_threads(1)

BASE_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger("laya-phishing")

CASCADE_THRESHOLD = 0.80
HIGH_RISK_THRESHOLD = 60.0
MEDIUM_RISK_THRESHOLD = 35.0
MAX_TEXT_LENGTH = 12_000
WARMUP_TEXT = (
    "hello dear customer. we updated our terms of service. "
    "please accept them by clicking this link: https://example.com/verify"
)

RiskLevel = Literal["low", "medium", "high"]
ModelStatus = Literal["loading", "ready", "error"]
Question = dict[str, Any]


class AnalyzeRequest(BaseModel):
    text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, max_length=MAX_TEXT_LENGTH),
    ]

    @field_validator("text")
    @classmethod
    def _require_text(cls, value: str) -> str:
        if not value:
            raise PydanticCustomError("empty_email", "Paste an email to analyze.")
        return value


class AnalyzeResponse(BaseModel):
    risk_score: float = Field(ge=0, le=100)
    level: RiskLevel
    seconds: float = Field(ge=0)


class StatusResponse(BaseModel):
    state: ModelStatus
    ready: bool
    error: str | None = None


def _choice_question(instructions: str, yes: str, no: str) -> Question:
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {"yes": yes, "no": no},
    }


GENERAL_QUESTION = _choice_question(
    "Is this email a phishing, scam, or social-engineering attempt?",
    "Yes. It uses deception, impersonation, pressure, or an unsafe request.",
    "No. It is an ordinary email with no deceptive request.",
)
BEC_QUESTION = _choice_question(
    "Is this a business email compromise or social-engineering attempt, such as "
    "an executive or supplier asking for gift cards, a wire transfer, secrecy, "
    "or a change in bank details?",
    "Yes, it is business email compromise, invoice fraud, or social engineering.",
    "No, it does not have that pattern.",
)


@dataclass
class ModelState:
    status: ModelStatus = "loading"
    router: Any | None = None
    error: str = ""


STATE = ModelState()
STATE_LOCK = threading.Lock()
INFERENCE_LOCK = asyncio.Lock()


def score_question(router: Any, key: str, question: Question, text: str) -> float:
    """Run one choice question and return its yes-probability (0..1)."""
    with torch.inference_mode():
        result = router.predict(
            {"body": text},
            {key: question},
            model="english",
            lang="en",
        )
    return float(result["answers"][key]["probabilities"]["yes"])


def risk_level(risk: float) -> RiskLevel:
    if risk >= HIGH_RISK_THRESHOLD:
        return "high"
    if risk >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def warm_up(router: Any) -> None:
    score_question(router, "general", GENERAL_QUESTION, WARMUP_TEXT)
    score_question(router, "bec", BEC_QUESTION, WARMUP_TEXT)


def load_model() -> None:
    try:
        # Pin CPU so the tested memory and thread settings stay predictable on
        # machines that also have a GPU.
        router = Router(max_loaded=2, device="cpu")
        router.preload(["english"])
        warm_up(router)
    except Exception as exc:
        LOGGER.exception("Laya failed to load")
        with STATE_LOCK:
            STATE.status = "error"
            STATE.error = f"{type(exc).__name__}: {exc}"
        return

    with STATE_LOCK:
        STATE.router = router
        STATE.status = "ready"
        STATE.error = ""
    LOGGER.info("Laya ready")


def analyze_text(router: Any, text: str) -> AnalyzeResponse:
    started = time.perf_counter()
    score = score_question(router, "general", GENERAL_QUESTION, text)
    if score < CASCADE_THRESHOLD:
        score = max(score, score_question(router, "bec", BEC_QUESTION, text))

    risk = round(score * 100, 2)
    return AnalyzeResponse(
        risk_score=risk,
        level=risk_level(risk),
        seconds=round(time.perf_counter() - started, 1),
    )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    threading.Thread(target=load_model, daemon=True, name="laya-loader").start()
    yield


app = FastAPI(
    title="Laya Phishing Detection",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        BASE_DIR / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def model_status() -> StatusResponse:
    with STATE_LOCK:
        status = STATE.status
        error = STATE.error
    return StatusResponse(
        state=status,
        ready=status == "ready",
        error=error or None,
    )


@app.post("/api/analyses")
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    text = payload.text

    with STATE_LOCK:
        status = STATE.status
        router = STATE.router
        error = STATE.error
    if status != "ready" or router is None:
        if status == "error":
            detail = f"Laya failed to load: {error or 'unknown error'}"
        else:
            detail = "Laya is still loading."
        raise HTTPException(status_code=503, detail=detail)

    try:
        async with INFERENCE_LOCK:
            return await asyncio.to_thread(analyze_text, router, text)
    except Exception:
        LOGGER.exception("Analysis failed")
        raise HTTPException(status_code=500, detail="The analysis failed.") from None


def _ensure_port_available(host: str, port: int) -> None:
    """Fail early with a clear message when the server port cannot be bound."""
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SystemExit(f"HOST {host!r} cannot be resolved: {exc}") from None

    family, socktype, protocol, _, address = addresses[0]
    with socket.socket(family, socktype, protocol) as probe:
        try:
            # A plain bind is used on purpose: SO_REUSEADDR can succeed on
            # Windows even while another process is listening on the port.
            probe.bind(address)
        except OSError:
            raise SystemExit(
                f"Cannot bind to {host}:{port}. Another program is already "
                "using that port, or it is reserved.\n"
                "Stop the other process, or choose a different port:\n"
                '  PowerShell:   $env:PORT="8001"; py -3.11 main.py\n'
                "  macOS/Linux:  PORT=8001 python3.11 main.py"
            ) from None


def _port_from_env() -> int:
    raw = os.getenv("PORT", "8000")
    try:
        port = int(raw)
    except ValueError:
        raise SystemExit(
            f"PORT must be an integer between 1 and 65535, got {raw!r}."
        ) from None
    if not 1 <= port <= 65535:
        raise SystemExit(f"PORT must be between 1 and 65535, got {port}.")
    return port


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = _port_from_env()
    _ensure_port_available(host, port)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
