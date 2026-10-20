"""Recover the 2048 -> 1024 projection used when features.pt was extracted.

Person A's features are  features = ResNet50_pooled(image) @ W.T + b  with a
randomly initialised, never saved (W, b). Because the map is linear, we can
recompute the pooled ResNet-50 features for a few thousand train2014 images,
pair them with the stored features, and solve for W and b by least squares.

The script prints a validation error on images NOT used for the fit. A relative
error near 0 (about 1e-4 or smaller) means the recovery worked. A large error
means the stored features were produced differently (other ResNet weights,
other preprocessing, extra layers) and the demo cannot reproduce them.

Usage (Colab):
  python app/recover_projection.py \
      --zip "/content/drive/MyDrive/VQA Project/dataset/train/train2014.zip" \
      --features data_local/features.pt --n 4000 --out projection.pt
"""
import argparse
import io
import random
import zipfile

import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True, help="Path to train2014.zip")
    p.add_argument("--features", required=True, help="Person A's features.pt")
    p.add_argument("--n", type=int, default=4000, help="Images used for the fit (>2049)")
    p.add_argument("--n_val", type=int, default=500, help="Extra images for checking the fit")
    p.add_argument("--out", default="projection.pt")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load(a.features, map_location="cpu")
    index = {name: i for i, name in enumerate(data["image_names"])}
    stored = data["features"]

    # Same backbone and preprocessing as Person A's script
    # (resnet50(pretrained=True) = the original ImageNet weights, IMAGENET1K_V1).
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = torch.nn.Identity()          # output = pooled 2048-d features
    resnet = resnet.to(device).eval()
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    zf = zipfile.ZipFile(a.zip)
    members = [m for m in zf.namelist()
               if m.endswith(".jpg") and m.split("/")[-1] in index]
    random.Random(0).shuffle(members)
    members = members[: a.n + a.n_val]
    print("Using", len(members), "images")

    pooled, target = [], []
    for start in range(0, len(members), 64):
        chunk = members[start:start + 64]
        imgs = [tf(Image.open(io.BytesIO(zf.read(m))).convert("RGB")) for m in chunk]
        with torch.no_grad():
            r = resnet(torch.stack(imgs).to(device)).cpu()
        pooled.append(r)
        target.append(stored[[index[m.split("/")[-1]] for m in chunk]])
        if (start // 64) % 10 == 0:
            print("  processed", start + len(chunk))
    R = torch.cat(pooled).double()           # [N, 2048]
    Y = torch.cat(target).double()           # [N, 1024]

    R_fit, Y_fit = R[: a.n], Y[: a.n]
    R_val, Y_val = R[a.n:], Y[a.n:]

    # Solve [R, 1] @ [W.T; b] = Y in the least-squares sense
    X = torch.cat([R_fit, torch.ones(len(R_fit), 1, dtype=R.dtype)], dim=1)
    sol = torch.linalg.lstsq(X, Y_fit).solution          # [2049, 1024]
    weight = sol[:-1].T.contiguous()                      # [1024, 2048]
    bias = sol[-1].contiguous()                           # [1024]

    pred = R_val @ weight.T + bias
    rel_err = ((pred - Y_val).norm() / Y_val.norm()).item()
    max_err = (pred - Y_val).abs().max().item()
    print("Validation relative error:", rel_err)
    print("Validation max abs error :", max_err)
    if rel_err > 1e-2:
        print("WARNING: large error. The stored features were probably produced "
              "differently, so the demo cannot reproduce them.")

    torch.save({"weight": weight.float(), "bias": bias.float(),
                "val_relative_error": rel_err}, a.out)
    print("Saved projection to", a.out)


if __name__ == "__main__":
    main()
