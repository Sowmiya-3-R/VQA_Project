"""FastAPI server for the VQA web demo.

It does two things:
  1. serves the web page in app/static/ (index.html, styles.css, script.js)
  2. exposes POST /api/predict, which runs the trained baseline model

Image pipeline (must match how the training features were made):
  image -> ResNet-50 (original ImageNet weights, pooled 2048-d)
        -> recovered projection 2048 -> 1024 (projection.pt)
        -> VQAModel together with the question token ids
        -> softmax -> top-5 answers

Run from the repo root:
  python app/server.py --checkpoint best_model.pt --vocab vocab.json \
      --answers answers.json --projection projection.pt
Then open http://127.0.0.1:8000
"""
import argparse
import io
import json
import os
import sys
import threading

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # repo root, so models/ and text_encoder/ import
from models.vqa_model import VQAModel
from text_encoder.tokenizer import encode_question  # check the name: grep -n "def " text_encoder/tokenizer.py

STATIC_DIR = os.path.join(HERE, "static")
MAX_LEN = 20                       # question length the model was trained with
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # same 10MB limit as the web page
MAX_QUESTION_CHARS = 240            # same limit as the web page
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def load_pipeline(args):
    """Load everything once at startup and return it in a dictionary."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = json.load(open(args.vocab, encoding="utf-8"))
    answers = json.load(open(args.answers, encoding="utf-8"))

    # Recovered projection (see recover_projection.py): features = pooled @ W.T + b
    projection = torch.load(args.projection, map_location="cpu")

    # Same backbone and preprocessing as the feature extraction used for training
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = torch.nn.Identity()  # output = pooled 2048-d vector
    resnet = resnet.to(device).eval()
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # The trained VQA model
    model = VQAModel(vocab_size=len(vocab), image_dim=1024, num_classes=len(answers))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    print(f"Loaded: device={device}, vocab={len(vocab)}, answers={len(answers)}")
    return {"device": device, "vocab": vocab, "answers": answers,
            "weight": projection["weight"], "bias": projection["bias"],
            "resnet": resnet, "transform": transform, "model": model}


def to_rgb(img):
    """Convert any image to RGB; transparent areas become white."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def run_vqa(state, img, question):
    """Return the top-5 (answer, probability) pairs for one image and question."""
    with torch.no_grad():
        x = state["transform"](img).unsqueeze(0).to(state["device"])
        pooled = state["resnet"](x).cpu()                       # [1, 2048]
        features = pooled @ state["weight"].T + state["bias"]   # [1, 1024]
        ids = torch.as_tensor(encode_question(question, state["vocab"], MAX_LEN),
                              dtype=torch.long).view(1, -1)     # [1, 20]
        logits = state["model"](features.to(state["device"]), ids.to(state["device"]))
        probs = F.softmax(logits, dim=1)[0].cpu()
        top = probs.topk(5)
    return [(state["answers"][int(i)], float(p)) for p, i in zip(top.values, top.indices)]


def create_app(args):
    state = load_pipeline(args)
    lock = threading.Lock()  # one prediction at a time keeps memory use predictable
    app = FastAPI(title="VQA demo")

    @app.post("/api/predict")
    def predict(image: UploadFile = File(...), question: str = Form(...)):
        question = question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Please type a question.")
        if len(question) > MAX_QUESTION_CHARS:
            raise HTTPException(status_code=400,
                                detail=f"Questions can be at most {MAX_QUESTION_CHARS} characters.")
        data = image.file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="That image is larger than 10MB.")
        try:
            img = Image.open(io.BytesIO(data))
            image_format = img.format
            img = to_rgb(img)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(status_code=400, detail="That file could not be read as an image.")
        if image_format not in ALLOWED_FORMATS:
            raise HTTPException(status_code=400, detail="Please upload a JPG, PNG or WEBP image.")

        with lock:
            top = run_vqa(state, img, question)
        return {"answer": top[0][0], "confidence": top[0][1],
                "alternatives": [{"answer": a, "confidence": p} for a, p in top]}

    # Serve the web page last, so /api/predict takes priority over the static files
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
