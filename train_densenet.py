import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model_baseline import build_baseline
from model_densenet import build_densenet
from model_densenet_manual import build_densenet_manual


LABELS = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}


def read_samples(root, seed, max_studies, split_file=None):
    labels = pd.read_csv(root / "train.csv").melt(
        id_vars="study_id", var_name="target", value_name="severity"
    )
    coords = pd.read_csv(root / "train_label_coordinates.csv")
    coords["target"] = (
        coords.condition.str.lower().str.replace(" ", "_", regex=False)
        + "_" + coords.level.str.lower().str.replace("/", "_", regex=False)
    )
    samples = coords.merge(labels, on=["study_id", "target"], validate="many_to_one")
    missing = samples.severity.isna().sum()
    samples = samples.dropna(subset=["severity"]).copy()
    samples["label"] = samples.severity.map(LABELS)
    if samples.label.isna().any():
        raise ValueError("Unknown severity in train.csv")
    samples["label"] = samples.label.astype(int)
    studies = sorted(samples.study_id.unique())
    if max_studies:
        studies = sorted(random.Random(seed).sample(studies, min(max_studies, len(studies))))
        samples = samples[samples.study_id.isin(studies)].copy()
    if len(studies) < 4:
        raise ValueError("Need at least 4 studies for train/validation/test")
    if split_file is not None and split_file.exists():
        split = pd.read_csv(split_file)
        if set(split.columns) != {"study_id", "split"} or split.study_id.duplicated().any():
            raise ValueError("Split file must contain one study_id and split per study")
        if set(split.study_id) != set(studies) or set(split.split) != {"train", "val", "test"}:
            raise ValueError("Split does not match this dataset/subset. Use the same data and max-studies.")
    else:
        train_ids, remaining_ids = train_test_split(studies, test_size=0.3, random_state=seed)
        val_ids, test_ids = train_test_split(remaining_ids, test_size=0.5, random_state=seed)
        split = pd.DataFrame({"study_id": studies, "split": "train"})
        split.loc[split.study_id.isin(val_ids), "split"] = "val"
        split.loc[split.study_id.isin(test_ids), "split"] = "test"
        if split_file is not None:
            split_file.parent.mkdir(parents=True, exist_ok=True)
            split.to_csv(split_file, index=False)
    samples = samples.merge(split, on="study_id", validate="many_to_one")
    samples["path"] = (
        "train_images/" + samples.study_id.astype(str) + "/"
        + samples.series_id.astype(str) + "/" + samples.instance_number.astype(str) + ".dcm"
    )
    for path in samples.path.unique():
        if not (root / path).is_file():
            raise FileNotFoundError(root / path)
    print(f"Skipped {missing} missing labels; studies per split: {split.split.value_counts().to_dict()}")
    return samples.reset_index(drop=True)


def crop_image(pixels, x, y, size):
    height, width = pixels.shape
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Coordinate ({x}, {y}) outside image {width}x{height}")
    x, y = min(round(x), width - 1), min(round(y), height - 1)
    # Them vien den de crop o sat mep van dung kich thuoc va tam.
    half = size // 2
    pixels = F.pad(pixels, (half, half, half, half))
    return pixels[y:y + size, x:x + size]


class SpineCrops(Dataset):
    def __init__(self, rows, root, crop_size, image_size):
        self.rows = rows.to_dict("records")
        self.root = root
        self.crop_size = crop_size
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        dicom = pydicom.dcmread(self.root / row["path"])
        pixels = torch.tensor(dicom.pixel_array.astype(np.float32))
        pixels = pixels * float(dicom.get("RescaleSlope", 1)) + float(dicom.get("RescaleIntercept", 0))
        pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min()).clamp_min(1e-6)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            pixels = 1 - pixels
        crop = crop_image(pixels, row["x"], row["y"], self.crop_size)
        image = F.interpolate(crop[None, None], size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)[0]
        return image, row["label"]


def make_model(name, pretrained=False):
    if name == "baseline":
        return build_baseline()
    if name == "densenet_manual":
        return build_densenet_manual()
    if name == "densenet":
        return build_densenet(pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")


def evaluate(model, loader, device):
    model.eval()
    probabilities, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities.append(model(images.to(device)).softmax(1).cpu().numpy())
            targets.extend(labels.tolist())
    probabilities = np.concatenate(probabilities).astype(np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(1)
    metrics = {
        "log_loss": log_loss(targets, probabilities, labels=[0, 1, 2]),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, labels=[0, 1, 2], average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1, 2]).tolist(),
        "class_counts": np.bincount(targets, minlength=3).tolist(),
        "crops": len(targets),
    }
    return metrics, probabilities


def save_results(output, metrics, probabilities, rows, split_name):
    metrics = dict(metrics, split=split_name)
    (output / f"{split_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions = rows.copy()
    predictions[["normal_mild", "moderate", "severe"]] = probabilities
    predictions.to_csv(output / f"{split_name}_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train DenseNet or the baseline on RSNA crops")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", choices=["baseline", "densenet_manual", "densenet"], default="densenet")
    parser.add_argument("--output", type=Path, default=Path("runs/densenet"))
    parser.add_argument("--evaluate", type=Path, help="Run directory containing best.pt")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Dataset for --evaluate")
    parser.add_argument("--split-file", type=Path, default=Path("runs/split.csv"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-studies", type=int, default=0, help="0 uses all studies")
    parser.add_argument("--pretrained", action="store_true", help="Optional ImageNet weights for library DenseNet only")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.crop_size < 2 or args.image_size < 64:
        parser.error("epochs >= 1, batch-size >= 2, crop-size >= 2 and image-size >= 64 required")
    if args.lr <= 0 or args.workers < 0 or args.max_studies < 0:
        parser.error("lr > 0, workers >= 0 and max-studies >= 0 required")
    if args.pretrained and args.model != "densenet":
        parser.error("--pretrained is only available for the library densenet")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.evaluate:
        output = args.evaluate
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if config.get("pipeline_version") != 2:
            raise ValueError("Old checkpoint uses a different pipeline. Retrain all models with the new 3-way split.")
        samples = pd.read_csv(output / "samples.csv")
        rows = samples[samples.split == args.split]
        if rows.empty:
            raise ValueError(f"No {args.split} samples saved with this checkpoint")
        dataset = SpineCrops(rows, args.data_root, config["crop_size"], config["image_size"])
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
        model = make_model(config["model"], pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        metrics, probabilities = evaluate(model, loader, device)
        metrics["epoch"] = checkpoint["epoch"]
        save_results(output, metrics, probabilities, rows, args.split)
        return

    samples = read_samples(args.data_root, args.seed, args.max_studies, args.split_file)
    train_rows = samples[samples.split == "train"]
    val_rows = samples[samples.split == "val"]
    if len(train_rows) < 2 or val_rows.empty:
        raise ValueError("Need at least 2 train crops and 1 validation crop")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {args.output}")
    config = vars(args).copy()
    config.update(data_root=str(args.data_root), output=str(args.output), evaluate=None,
                  split_file=str(args.split_file), pipeline_version=2,
                  torch_version=str(torch.__version__), device=device)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    samples.to_csv(args.output / "samples.csv", index=False)
    train_data = SpineCrops(train_rows, args.data_root, args.crop_size, args.image_size)
    val_data = SpineCrops(val_rows, args.data_root, args.crop_size, args.image_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed),
                              num_workers=args.workers, pin_memory=device == "cuda")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers)
    model = make_model(args.model, config["pretrained"]).to(device)
    config["parameters"] = sum(p.numel() for p in model.parameters())
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best_loss = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total += len(labels)
            if step % 100 == 0:
                print(f"Epoch {epoch}, batch {step}/{len(train_loader)}", flush=True)
        metrics, probabilities = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        history.append({"epoch": epoch, "train_loss": total_loss / total,
                        "val_loss": metrics["log_loss"], "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"]})
        pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
        print(history[-1], flush=True)
        if metrics["log_loss"] < best_loss:
            best_loss = metrics["log_loss"]
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch},
                       args.output / "best.pt")
            save_results(args.output, metrics, probabilities, val_rows, "val")


if __name__ == "__main__":
    main()
