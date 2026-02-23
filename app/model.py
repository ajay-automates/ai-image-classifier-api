"""
Model Inference Engine
Loads CLIP model once at startup, runs inference on uploaded images.
No external API calls - everything runs locally.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import time
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CATEGORIES = [
    "a photo of a dog", "a photo of a cat", "a photo of a car",
    "a photo of a person", "a photo of food", "a photo of a building",
    "a photo of a landscape", "a photo of electronics", "a photo of furniture",
    "a photo of clothing", "a photo of an animal", "a photo of a flower",
    "a photo of a book", "a photo of a phone", "a photo of a computer",
    "a screenshot of a website", "a photo of art", "a medical image",
    "a photo of a document", "a photo of a chart or graph",
]


class ImageClassifier:
    """Self-hosted image classifier using CLIP."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = DEVICE
        self.model_name = MODEL_NAME
        self.loaded = False
        self.load_time_ms = 0

    def load(self):
        """Load model into memory. Called once at startup."""
        start = time.time()
        logger.info(f"Loading {self.model_name} on {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.load_time_ms = (time.time() - start) * 1000
        self.loaded = True
        logger.info(f"Model loaded in {self.load_time_ms:.0f}ms on {self.device}")

    def classify(self, image: Image.Image, categories: list = None) -> dict:
        """Classify an image into categories using zero-shot CLIP."""
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if categories is None:
            categories = DEFAULT_CATEGORIES

        start = time.time()
        inputs = self.processor(text=categories, images=image, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=0).cpu().numpy()
        latency_ms = (time.time() - start) * 1000

        results = []
        for cat, prob in zip(categories, probs):
            label = cat.replace("a photo of ", "").replace("a ", "").strip()
            results.append({"label": label, "score": round(float(prob), 4), "rank": 0})

        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return {
            "top_prediction": results[0]["label"],
            "confidence": results[0]["score"],
            "predictions": results[:5],
            "all_predictions": results,
            "num_categories": len(categories),
            "latency_ms": round(latency_ms, 1),
            "device": self.device,
            "model": self.model_name,
        }

    def classify_custom(self, image: Image.Image, labels: list) -> dict:
        """Classify with custom user-provided labels."""
        categories = [f"a photo of {label}" for label in labels]
        result = self.classify(image, categories)
        for pred in result["all_predictions"]:
            for label in labels:
                if label in pred["label"]:
                    pred["label"] = label
        result["top_prediction"] = result["predictions"][0]["label"]
        return result

    def get_status(self) -> dict:
        return {
            "loaded": self.loaded,
            "model": self.model_name,
            "device": self.device,
            "load_time_ms": round(self.load_time_ms, 1),
            "gpu_available": torch.cuda.is_available(),
        }


classifier = ImageClassifier()
