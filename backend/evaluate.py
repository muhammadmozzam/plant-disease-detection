"""Measured evaluation of the three saved models.

1. Lab test: the PlantVillage validation folder (20% split, never used for
   training) -> accuracy, macro-F1, per-class report, confusion matrix.
2. Field test: PlantDoc photos taken in real fields, mapped to the matching
   PlantVillage classes -> the same metrics, so lab and field can be compared.
3. Cost: CPU time per image, parameter count and file size.

Example (Kaggle):
  python backend/evaluate.py --weights /kaggle/input/plant-models \
      --lab /kaggle/input/plantvillage/PlantVillage/val \
      --field /kaggle/input/plantdoc-dataset --out results
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

import models as M
from plantdoc_map import match_plantdoc_folder

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ImageList(Dataset):
    def __init__(self, items):
        self.items = items  # list of (path, label_index)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        return M.TRANSFORM(img), y


def list_images(folder):
    out = []
    for root, _, files in os.walk(folder):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(IMG_EXT)]
    return sorted(out)


def lab_items(lab_dir, classes):
    items = []
    for i, c in enumerate(classes):
        d = os.path.join(lab_dir, c)
        if os.path.isdir(d):
            items += [(p, i) for p in list_images(d)]
    return items


def field_items(field_dir, classes):
    """Every PlantDoc class folder (train and test parts) that maps to a PlantVillage class."""
    items, used, skipped = [], {}, set()
    for root, dirs, _ in os.walk(field_dir):
        for d in dirs:
            target = match_plantdoc_folder(d)
            if target is None:
                p = os.path.join(root, d)
                if any(f.lower().endswith(IMG_EXT) for f in os.listdir(p)):  # a class folder, not train/ or test/
                    skipped.add(d)
                continue
            imgs = [p for p in list_images(os.path.join(root, d))]
            items += [(p, classes.index(target)) for p in imgs]
            used[d] = used.get(d, 0) + len(imgs)
    return items, used, sorted(skipped)


@torch.no_grad()
def predict(model, items, device, batch_size=64, workers=2):
    dl = DataLoader(ImageList(items), batch_size=batch_size, num_workers=workers)
    logits, ys = [], []
    for x, y in dl:
        logits.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def scores(logits, y, allowed=None):
    """allowed: class indices the prediction may choose from (None = all 38)."""
    z = logits.copy()
    if allowed is not None:
        mask = np.full(z.shape[1], -np.inf)
        mask[allowed] = 0
        z = z + mask
    pred = z.argmax(1)
    top3 = np.argsort(-z, 1)[:, :3]
    return pred, {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0)),
        "top3_accuracy": float(np.mean([y[i] in top3[i] for i in range(len(y))])),
    }


@torch.no_grad()
def cpu_speed(model, n=100):
    model = model.cpu().eval()
    x = torch.rand(1, 3, 224, 224)
    for _ in range(10):
        model(x)
    t = time.perf_counter()
    for _ in range(n):
        model(x)
    return (time.perf_counter() - t) / n * 1000


def save_confusion(y, pred, classes, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y, pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    short = [classes[i].replace("___", ": ").replace("_", " ")[:32] for i in labels]
    ax.set_xticks(range(len(labels)), short, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), short, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="folder with the three .pth files")
    ap.add_argument("--lab", required=True, help="PlantVillage val folder")
    ap.add_argument("--field", default=None, help="PlantDoc folder (optional)")
    ap.add_argument("--classes", default=os.path.join(os.path.dirname(__file__), "classes.json"))
    ap.add_argument("--models", default="resnet18,cnn,mobilenetv2")
    ap.add_argument("--out", default="results")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = M.class_names(json_path=args.classes)

    lab = lab_items(args.lab, classes)
    print(f"Lab test: {len(lab)} images in {len({y for _, y in lab})} classes")
    field, used, skipped = ([], {}, [])
    if args.field:
        field, used, skipped = field_items(args.field, classes)
        print(f"Field test: {len(field)} images; used folders: {used}")
        print(f"PlantDoc folders with no PlantVillage match (left out): {skipped}")
        with open(os.path.join(args.out, "plantdoc_mapping_used.json"), "w") as f:
            json.dump({"used": used, "left_out": skipped}, f, indent=1)

    shared = sorted({y for _, y in field}) if field else None
    rows = []
    for name in args.models.split(","):
        model = M.load(name, args.weights, len(classes), device)
        params = sum(p.numel() for p in model.parameters())
        size_mb = os.path.getsize(os.path.join(args.weights, M.WEIGHT_FILES[name])) / 2**20

        lg, y = predict(model, lab, device, args.batch_size, args.workers)
        pred, s = scores(lg, y)
        rows.append({"model": name, "test": "lab (PlantVillage val, 38 classes)", **s})
        pd.DataFrame(classification_report(y, pred, labels=range(len(classes)), target_names=classes,
                                           output_dict=True, zero_division=0)).T.to_csv(
            os.path.join(args.out, f"{name}_lab_per_class.csv"))
        save_confusion(y, pred, classes, list(range(len(classes))),
                       os.path.join(args.out, f"{name}_lab_confusion.png"), f"{name}: lab test")

        if field:
            # Same classes in both tests, so the lab-vs-field gap is a fair comparison.
            keep = np.isin(y, shared)
            _, s = scores(lg[keep], y[keep], allowed=shared)
            rows.append({"model": name, "test": f"lab, shared {len(shared)} classes", **s})
            flg, fy = predict(model, field, device, args.batch_size, args.workers)
            _, s = scores(flg, fy)
            rows.append({"model": name, "test": "field (PlantDoc), any of 38 classes", **s})
            fpred, s = scores(flg, fy, allowed=shared)
            rows.append({"model": name, "test": f"field (PlantDoc), shared {len(shared)} classes", **s})
            pd.DataFrame(classification_report(fy, fpred, labels=shared, target_names=[classes[i] for i in shared],
                                               output_dict=True, zero_division=0)).T.to_csv(
                os.path.join(args.out, f"{name}_field_per_class.csv"))
            save_confusion(fy, fpred, classes, shared,
                           os.path.join(args.out, f"{name}_field_confusion.png"), f"{name}: field test")
            pd.DataFrame({"path": [p for p, _ in field], "true": [classes[i] for i in fy],
                          "pred": [classes[i] for i in fpred]}).to_csv(
                os.path.join(args.out, f"{name}_field_predictions.csv"), index=False)

        ms = round(cpu_speed(model), 1)
        for r in rows:
            if r["model"] == name:
                r.update(params_millions=round(params / 1e6, 2), file_mb=round(size_mb, 1),
                         cpu_ms_per_image=ms)
        print(pd.DataFrame([r for r in rows if r["model"] == name]).to_string(index=False))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
