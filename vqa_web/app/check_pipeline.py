"""Check that the demo's image pipeline reproduces the stored training features.

For an image that is inside features.pt, this computes
    ResNet-50 (ImageNet V1 weights, pooled 2048-d) @ projection.weight.T + projection.bias
and compares it with the vector stored in features.pt for the same image.

Usage (from the repo root, on CPU):
  python app/check_pipeline.py --features features.pt --projection projection.pt \
      --image COCO_train2014_000000000009.jpg

The image file name must be one of the names stored in features.pt.
A relative error below 1e-3 means the pipeline is faithful. A larger error means
the demo would feed the model features it was not trained on, so do not use it
to answer uploaded photos.
"""
import argparse
import os

import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="features.pt from Person A")
    parser.add_argument("--projection", required=True, help="projection.pt (weight, bias)")
    parser.add_argument("--image", required=True, nargs="+",
                        help="one or more COCO train2014 image files (names must be in features.pt)")
    args = parser.parse_args()

    data = torch.load(args.features, map_location="cpu")
    names = data["image_names"]
    position = {name: i for i, name in enumerate(names)}
    projection = torch.load(args.projection, map_location="cpu")

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    worst = 0.0
    for path in args.image:
        name = os.path.basename(path)
        if name not in position:
            print(f"{name}: not found in features.pt, skipped")
            continue
        stored = data["features"][position[name]]
        image = Image.open(path).convert("RGB")
        with torch.no_grad():
            pooled = resnet(transform(image).unsqueeze(0))[0]
        computed = pooled @ projection["weight"].T + projection["bias"]
        relative = ((computed - stored).norm() / stored.norm()).item()
        cosine = torch.nn.functional.cosine_similarity(computed, stored, dim=0).item()
        worst = max(worst, relative)
        print(f"{name}: relative error {relative:.6f}, cosine similarity {cosine:.6f}")

    print("RESULT:", "pipeline matches the stored features" if worst < 1e-3
          else "MISMATCH. Do not use this projection or pipeline for uploaded photos.")


if __name__ == "__main__":
    main()
