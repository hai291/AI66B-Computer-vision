"""Train/evaluate ResNet-18 tren ANH THAT (issue #4).

Chia theo study (train/val khong lan slice cua nhau), loss = CrossEntropy
co class-weight tu phan bo train, metric: loss + acc tren ca 2 tap.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from resnet_model import resnet18, count_parameters
from spine_dataset import SpineCropDataset, build_samples

CLASS_NAMES = ["Normal/Mild", "Moderate", "Severe"]
EPOCHS = 8
BATCH = 16


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = build_samples()
    studies = sorted(df.study_id.unique())
    cut = int(len(studies) * 0.75)  # 9 train / 3 val tren 12 studies
    train_df = df[df.study_id.isin(studies[:cut])]
    val_df = df[df.study_id.isin(studies[cut:])]
    print(f"train: {len(train_df)} crops / val: {len(val_df)} crops")

    train_loader = DataLoader(SpineCropDataset(train_df), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(SpineCropDataset(val_df), batch_size=BATCH)

    freq = train_df.label.value_counts(normalize=True).sort_index()
    print("phan bo train: " + ", ".join(f"{CLASS_NAMES[i]}={freq[i]:.1%}" for i in range(3)))
    weights = torch.tensor([(1.0 / freq[i]) for i in range(3)], dtype=torch.float32)
    weights = (weights / weights.mean()).to(device)

    model = resnet18(in_channels=1, num_classes=3).to(device)
    print(f"ResNet-18 params: {count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, False)
        print(f"Epoch {epoch+1}/{EPOCHS} - train loss: {tr_loss:.4f} acc: {tr_acc:.2%} | "
              f"val loss: {va_loss:.4f} acc: {va_acc:.2%}")


if __name__ == "__main__":
    main()
