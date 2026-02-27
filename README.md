<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=18,24,30&height=170&section=header&text=AI%20Image%20Classifier&fontSize=48&fontAlignY=35&animation=twinkling&fontColor=ffffff&desc=Self-Hosted%20CLIP%20Inference%20%7C%20%240%20Per%20Request&descAlignY=55&descSize=18" width="100%" />

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](.)
[![CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991?style=for-the-badge&logo=openai&logoColor=white)](.)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](.)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployed-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](.)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Upload any image. Get instant classification. The model runs inside your container — zero API costs.**

[Endpoints](#endpoints) · [Architecture](#architecture) · [Quick Start](#run-locally) · [Cost Comparison](#cost-comparison)

</div>

---

## Why This Exists

Most AI projects call external APIs (OpenAI, Claude, etc.) for every request. This project runs the model **locally** — no API keys, no per-request costs, no external dependencies.

```
Typical AI app:  Your App ──→ Claude API ──→ Response (costs $0.003/req)
This project:    Your App ──→ YOUR Model ──→ Response (costs $0/req)
```

---

## Architecture

```
Client (curl / browser / app)
        │
        ▼
FastAPI Server (uvicorn)
        │
        ▼
CLIP Model (loaded in memory at startup)
        │
        ▼
PyTorch Inference (CPU or GPU)
        │
        ▼
JSON Response with predictions + confidence scores
```

**Key design decisions:** Model loads ONCE at startup (not per-request). Model downloaded at Docker build time (fast cold starts). Zero external API calls during inference. Supports CPU and GPU automatically.

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/classify` | Classify image with default 20 categories |
| `POST` | `/classify/custom?labels=dog,cat,bird` | Classify with YOUR custom labels |
| `POST` | `/classify/batch` | Classify up to 10 images at once |
| `GET` | `/health` | Model status and health check |
| `GET` | `/metrics` | Request count and average latency |
| `GET` | `/docs` | Interactive Swagger UI |

### Example Response

```json
{
  "top_prediction": "dog",
  "confidence": 0.8734,
  "predictions": [
    {"label": "dog", "score": 0.8734, "rank": 1},
    {"label": "animal", "score": 0.0621, "rank": 2},
    {"label": "cat", "score": 0.0198, "rank": 3}
  ],
  "latency_ms": 45.2,
  "device": "cpu",
  "model": "openai/clip-vit-base-patch32"
}
```

---

## Run Locally

```bash
git clone https://github.com/ajay-automates/ai-image-classifier-api.git
cd ai-image-classifier-api
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
# Open http://localhost:7860/docs
```

### With Docker

```bash
docker build -t image-classifier .
docker run -p 7860:7860 image-classifier
```

### Deploy on HuggingFace Spaces (Free)

1. Create a new Space → select Docker SDK
2. Connect your GitHub repo
3. Deploy — Dockerfile handles everything

---

## MLOps Features

| Feature | Implementation |
|---------|---------------|
| **Model Serving** | FastAPI + uvicorn |
| **Containerization** | Docker with model pre-download |
| **Health Checks** | `/health` endpoint |
| **Metrics** | `/metrics` — request count, avg latency |
| **Batch Inference** | `/classify/batch` — up to 10 images |
| **Zero-shot** | Custom labels at query time — no retraining |
| **Auto GPU/CPU** | Detects CUDA automatically |
| **CORS** | Enabled for frontend integration |
| **API Docs** | Swagger UI at `/docs` |

---

## Cost Comparison

| Approach | Cost per 1M requests |
|----------|---------------------|
| GPT-4 Vision API | ~$10,000 |
| Claude Vision API | ~$3,000 |
| **This project** | **$0** |

---

## Tech Stack

`FastAPI` `PyTorch` `CLIP` `Transformers` `Docker` `Uvicorn` `HuggingFace` `Model Serving` `MLOps`

---

## Related Projects

| Project | Description |
|---------|-------------|
| [Resume Analyzer (QLoRA)](https://github.com/ajay-automates/advanced-resume-analyzer-qlora) | Fine-tuned Gemma 3 4B for resume-job fit |
| [AI Code Review Bot](https://github.com/ajay-automates/ai-code-review-bot) | Automated PR reviews via Claude + GitHub Actions |
| [AI Support Agent](https://github.com/ajay-automates/ai-support-agent) | RAG chatbot with LangSmith observability |

---

<div align="center">

**Built by [Ajay Kumar Reddy Nelavetla](https://github.com/ajay-automates)** · February 2026

*Own your inference. Zero API costs. Self-hosted ML in production.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=18,24,30&height=100&section=footer" width="100%" />

</div>
