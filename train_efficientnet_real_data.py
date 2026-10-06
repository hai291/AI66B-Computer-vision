import argparse
import csv
from pathlib import Path

import pydicom
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from model import EfficientNetB0


LABEL_TO_INDEX = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def label_column(condition, level):
    return f"{condition.lower().replace(' ', '_')}_{level.lower().replace('/', '_')}"


class RealDicomCropDataset(Dataset):
    def __init__(self, data_root, samples=2, crop_size=128):
        self.data_root = Path(data_root)
        train_rows = {
            int(row["study_id"]): row
            for row in read_csv(self.data_root / "train.csv")
        }
        coordinates = read_csv(self.data_root / "train_label_coordinates.csv")
        self.records = []
        for coordinate in coordinates:
            study_id = int(coordinate["study_id"])
            instance = coordinate["instance_number"]
            dicom_path = (
                self.data_root
                / "train_images"
                / str(study_id)
                / coordinate["series_id"]
                / f"{instance}.dcm"
            )
            column = label_column(coordinate["condition"], coordinate["level"])
            severity = train_rows.get(study_id, {}).get(column)
            if dicom_path.exists() and severity in LABEL_TO_INDEX:
                self.records.append(
                    {
                        "path": dicom_path,
                        "x": float(coordinate["x"]),
                        "y": float(coordinate["y"]),
                        "label": LABEL_TO_INDEX[severity],
                        "severity": severity,
                    }
                )
            if len(self.records) >= samples:
                break
        if len(self.records) < samples:
            raise ValueError(
                f"Only found {len(self.records)} usable DICOM crops; requested {samples}."
            )
        self.crop_size = crop_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        pixels = pydicom.dcmread(record["path"]).pixel_array
        pixels = torch.as_tensor(pixels, dtype=torch.float32)
        pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min()).clamp_min(1e-6)

        center_x, center_y = round(record["x"]), round(record["y"])
        half_size = min(pixels.shape) // 4
        left = max(0, center_x - half_size)
        top = max(0, center_y - half_size)
        right = min(pixels.shape[1], center_x + half_size)
        bottom = min(pixels.shape[0], center_y + half_size)
        crop = pixels[top:bottom, left:right]
        crop = functional.interpolate(
            crop.unsqueeze(0).unsqueeze(0),
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return crop, torch.tensor(record["label"], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet on real DICOM crops.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            r"C:\Users\kient\OneDrive\Documents"
            r"\rsna-2024-lumbar-spine-degenerative-classification"
        ),
    )
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    dataset = RealDicomCropDataset(args.data_root, samples=args.samples)
    loader = DataLoader(dataset, batch_size=args.samples, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB0().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(args.epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    print(
        f"trained EfficientNetB0 on {len(dataset)} real DICOM crops "
        f"for {args.epochs} epoch(s); device={device}; loss={loss.item():.4f}"
    )


if __name__ == "__main__":
    main()
