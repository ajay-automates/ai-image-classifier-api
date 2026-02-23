"""
FastAPI Server - AI Image Classifier API
Self-hosted model inference. No external API calls.
"""

import io
import time
import logging
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.model import classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

request_count = 0
total_latency_ms = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading model...")
    classifier.load()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Image Classifier API",
    description="Self-hosted image classification using OpenAI CLIP. No external API calls.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def root():
    return {
        "name": "AI Image Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "POST /classify": "Classify an uploaded image",
            "POST /classify/custom": "Classify with custom labels",
            "POST /classify/batch": "Classify multiple images",
            "GET /health": "Health check",
            "GET /metrics": "Request metrics",
        },
        "model": classifier.get_status(),
    }


@app.get("/health")
def health():
    status = classifier.get_status()
    return {"status": "healthy" if status["loaded"] else "loading", "model": status}


@app.get("/metrics")
def metrics():
    avg = (total_latency_ms / request_count) if request_count > 0 else 0
    return {"total_requests": request_count, "average_latency_ms": round(avg, 1), "model": classifier.get_status()}


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Classify an uploaded image using default categories."""
    global request_count, total_latency_ms
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {str(e)}")
    result = classifier.classify(image)
    request_count += 1
    total_latency_ms += result["latency_ms"]
    return result


@app.post("/classify/custom")
async def classify_custom(
    file: UploadFile = File(...),
    labels: str = Query(..., description="Comma-separated labels, e.g.: dog,cat,bird"),
):
    """Classify with custom user-provided labels."""
    global request_count, total_latency_ms
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    label_list = [l.strip() for l in labels.split(",") if l.strip()]
    if len(label_list) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 labels")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {str(e)}")
    result = classifier.classify_custom(image, label_list)
    request_count += 1
    total_latency_ms += result["latency_ms"]
    return result


@app.post("/classify/batch")
async def classify_batch(files: list[UploadFile] = File(...)):
    """Classify multiple images at once (max 10)."""
    global request_count, total_latency_ms
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    results = []
    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            results.append({"filename": f.filename, "error": "Not an image"})
            continue
        try:
            contents = await f.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            result = classifier.classify(image)
            result["filename"] = f.filename
            results.append(result)
            request_count += 1
            total_latency_ms += result["latency_ms"]
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})
    return {"batch_size": len(files), "results": results}
