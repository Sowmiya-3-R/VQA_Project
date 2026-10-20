"""Gradio demo for the baseline VQA model.

Pipeline for an uploaded image:
  image -> ResNet-50 (pooled 2048-d) -> recovered projection (1024-d) -> VQAModel
question -> tokenizer -> VQAModel -> top-5 answers with confidence

Usage (Colab, from the repo folder):
  python app/app.py --checkpoint ".../checkpoints_full/best_model.pt" \
      --vocab data_local/vocab.json --answers data_local/answers.json \
      --projection projection.pt --share
"""
import argparse
import json
import os
import sys

import gradio as gr
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

# Make the repo root importable (this file lives in app/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vqa_model import VQAModel
from text_encoder.tokenizer import encode_question  # check this name with: grep -n "def " text_encoder/tokenizer.py

MAX_LEN = 20


def build_pipeline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = json.load(open(args.vocab, encoding="utf-8"))
    answers = json.load(open(args.answers, encoding="utf-8"))

    # Recovered projection (see recover_projection.py)
    proj = torch.load(args.projection, map_location="cpu")
    weight, bias = proj["weight"], proj["bias"]

    # Same backbone and preprocessing as the feature extraction
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device).eval()
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # VQA model with the trained weights
    model = VQAModel(vocab_size=len(vocab), image_dim=1024, num_classes=len(answers))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model = model.to(device).eval()
    return device, vocab, answers, weight, bias, resnet, transform, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--answers", required=True)
    p.add_argument("--projection", required=True)
    p.add_argument("--share", action="store_true", help="Create a public link (needed in Colab)")
    args = p.parse_args()
    device, vocab, answers, weight, bias, resnet, transform, model = build_pipeline(args)

    @torch.no_grad()
    def predict(image, question):
        if image is None or not question or not question.strip():
            raise gr.Error("Please upload an image and type a question.")
        x = transform(image.convert("RGB")).unsqueeze(0).to(device)
        pooled = resnet(x).cpu()                              # [1, 2048]
        feat = pooled @ weight.T + bias                       # [1, 1024]
        ids = torch.as_tensor(encode_question(question, vocab, MAX_LEN),
                              dtype=torch.long).view(1, -1)   # [1, 20]
        logits = model(feat.to(device), ids.to(device))       # [1, num_answers]
        probs = F.softmax(logits, dim=1)[0].cpu()
        top = probs.topk(5)
        return {answers[i]: float(s) for s, i in zip(top.values, top.indices)}

    demo = gr.Interface(
        fn=predict,
        inputs=[gr.Image(type="pil", label="Image"),
                gr.Textbox(label="Question", placeholder="e.g. What color is the car?")],
        outputs=gr.Label(num_top_classes=5, label="Answer (top 5 with confidence)"),
        title="Visual Question Answering (baseline)",
        description=("Baseline model: ResNet-50 image features + LSTM question encoder, "
                     "fused by element-wise multiplication, trained on VQA v2. "
                     "Answers come from the 3,000 most common answers."),
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()