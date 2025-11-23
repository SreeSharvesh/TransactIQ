# backend/app.py
import os
import json
import asyncio
from typing import Any, Dict, List
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .model import HybridTransactIQModel

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ---------------- Gemini config ----------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

try:
    import google.generativeai as genai
except ImportError:
    genai = None

gemini_model = None
if GEMINI_API_KEY and genai is not None:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# ---------------- FastAPI setup ----------------

app = FastAPI(title="TransactIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Serve /assets/* (JS, CSS, images)
app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, "assets")), name="assets")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the React app."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

model = HybridTransactIQModel()

# queue for "live" stream of /api/predict events
stream_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=1000)


def push_to_stream(event: Dict[str, Any]) -> None:
    try:
        stream_queue.put_nowait(event)
    except asyncio.QueueFull:
        pass


# ---------------- health ----------------

@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------- predict / batch ----------------

@app.post("/api/predict")
async def predict(payload: Dict[str, Any]):
    desc = payload.get("description", "")
    amount = float(payload.get("amount", 0.0))
    date = payload.get("date", "2024-01-01")
    user_name = payload.get("user_name")
    account_id = payload.get("account_id")

    t0 = time.time()
    result = model.predict(desc, amount, date, user_name, account_id)
    t1 = time.time()
    latency_ms = (t1 - t0) * 1000.0
    ts = time.time()

    event = {
        "description": desc,
        "amount": amount,
        "date": date,
        "prediction": result,
        "latency_ms": latency_ms,
        "ts": ts,
        "source": "live",
    }
    push_to_stream(event)

    # also return latency to caller
    return {
        **result,
        "latency_ms": latency_ms,
        "ts": ts,
    }


@app.post("/api/batch")
async def batch_predict(payload: Dict[str, Any]):
    txs: List[Dict[str, Any]] = payload.get("transactions", [])
    results: List[Dict[str, Any]] = []
    import time

    t0 = time.time()
    for tx in txs:
        res = model.predict(
            tx.get("description", ""),
            float(tx.get("amount", 0.0)),
            tx.get("date", "2024-01-01"),
            tx.get("user_name"),
            tx.get("account_id"),
        )
        results.append(res)
    total = time.time() - t0

    return {
        "results": results,
        "batch_size": len(txs),
        "total_time_s": total,
        "latency_ms": (total / max(len(txs), 1)) * 1000,
        "throughput_tx_per_sec": (len(txs) / total) if total > 0 else None,
    }


# ---------------- metrics / taxonomy ----------------

@app.get("/api/metrics")
def get_metrics():
    return model.get_metrics()


@app.get("/api/taxonomy")
def get_taxonomy():
    return model.get_taxonomy()


@app.post("/api/reload-taxonomy")
def reload_taxonomy():
    model.reload_taxonomy()
    return {"status": "ok"}


# ---------------- feedback ----------------

@app.post("/api/feedback")
def feedback(payload: Dict[str, Any]):
    model.log_feedback(payload)
    return {"status": "logged"}


# ---------------- benchmarks & bias ----------------

@app.post("/api/benchmark")
def benchmark():
    return model.run_benchmark(n=256)


@app.get("/api/bias-report")
def bias_report():
    return model.bias_report()


# ---------------- Gemini explanation / advice ------

@app.post("/api/explain-metrics")
def explain_metrics(payload: Dict[str, Any]):
    if gemini_model is None:
        raise HTTPException(
            status_code=400,
            detail="Gemini API not configured or google-generativeai not installed.",
        )
    metrics = payload.get("metrics")
    if metrics is None:
        raise HTTPException(status_code=400, detail="metrics not provided")

    prompt = (
        "You are helping evaluate a transaction categorisation model. "
        "Given this JSON metrics dump (macro F1, confusion matrix, per-head scores), "
        "write a short, practical analysis (3-6 bullet points) for an ML engineer.\n\n"
        f"JSON:\n{json.dumps(metrics)[:6000]}"
    )
    resp = gemini_model.generate_content(prompt)
    text = resp.text if hasattr(resp, "text") else str(resp)
    return {"explanation": text}


@app.post("/api/advice")
def advice(payload: Dict[str, Any]):
    """
    Generate budgeting / safety advice for a single transaction.
    Used from streaming tab.
    """
    if gemini_model is None:
        return {"advice": None, "note": "Gemini not configured."}

    desc = payload.get("description", "")
    amount = payload.get("amount", 0.0)
    category = payload.get("category", "")

    prompt = (
        "You are a personal finance assistant. "
        "Given a transaction description, amount, and category, "
        "give ONE short, concrete budgeting insight (max 2 sentences).\n\n"
        f"Description: {desc}\n"
        f"Amount: {amount}\n"
        f"Category: {category}\n"
    )
    resp = gemini_model.generate_content(prompt)
    text = resp.text if hasattr(resp, "text") else str(resp)
    return {"advice": text}


# ---------------- streaming endpoints --------------

@app.get("/api/stream")
async def stream_predictions():
    """
    SSE stream of live /api/predict events.
    """

    async def event_generator():
        while True:
            event = await stream_queue.get()
            data = json.dumps(event)
            yield f"data: {data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/stream-test")
async def stream_test_dataset():
    """
    SSE stream that replays TEST_SAMPLE_PATH with predictions and optional Gemini advice.
    """
    async def event_generator():
        for row in model.iter_test_stream():
            desc = row["transaction_description"]
            amount = row["amount"]
            date = row["date"]

            t0 = time.time()
            pred = model.predict(desc, amount, date)
            t1 = time.time()
            latency_ms = (t1 - t0) * 1000.0
            ts = time.time()

            advice_text = None
            if gemini_model is not None:
                try:
                    prompt = (
                        "Give one short budgeting or risk insight (1–2 sentences) "
                        "for this transaction.\n\n"
                        f"Description: {desc}\n"
                        f"Amount: {amount}\n"
                        f"Category: {pred['category']}\n"
                        f"Risk tier: {pred.get('risk_tier','unknown')}\n"
                    )
                    resp = gemini_model.generate_content(prompt)
                    advice_text = resp.text if hasattr(resp, "text") else str(resp)
                except Exception:
                    advice_text = None

            event = {
                "description": desc,
                "amount": amount,
                "date": date,
                "prediction": pred,
                "latency_ms": latency_ms,
                "ts": ts,
                "advice": advice_text,
                "source": "test",
            }
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0.15)  # small delay to look realtime

    return StreamingResponse(event_generator(), media_type="text/event-stream")

