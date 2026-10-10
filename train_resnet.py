import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from resnet_model import resnet18, count_parameters

CLASS_NAMES = ["Normal/Mild", "Moderate", "Severe"]
LABEL2IDX = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}


class FakeSpineDataset(Dataset):
    # Giong het train_demo.py: gia lap crop MRI + nhan 0/1/2
    def __init__(self, num_samples=300, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = torch.randint(0, 3, (1,)).item()
        img = torch.randn(1, self.img_size, self.img_size)
        c = self.img_size // 2
        if label == 0:  # Normal/Mild: lam toi vung giua
            img[:, c-4:c+4, c-4:c+4] -= 0.7
        elif label == 2:  # Severe: lam sang vung giua
            img[:, c-4:c+4, c-4:c+4] += 0.7
        return img, label


def load_class_weights(csv_path="data/train.csv"):
    """Dem phan bo that cua 25 cot nhan -> weight = 1 / tan suat (chuan hoa ve mean=1)."""
    try:
        import pandas as pd
    except ImportError:
        print("khong co pandas -> dung weight deu")
        return None
    try:
        df = pd.read_csv(csv_path)
    except (FileNotFoundError, PermissionError):
        print(f"khong thay {csv_path} -> dung weight deu")
        return None
    label_cols = [c for c in df.columns if c != "study_id"]
    counts = [(df[c].map(LABEL2IDX).value_counts()).reindex([0, 1, 2], fill_value=0)
              for c in label_cols]
    total = sum(c.sum() for c in counts)
    freq = [(sum(c[i] for c in counts) / total) for i in range(3)]
    weights = torch.tensor([(1.0 / f) for f in freq], dtype=torch.float32)
    weights = weights / weights.mean()  # giu loss cung scale voi baseline
    print(f"phan bo that {csv_path}: " +
          ", ".join(f"{n}={freq[i]:.1%}" for i, n in enumerate(CLASS_NAMES)))
    print(f"class weights: {weights.tolist()}")
    return weights


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = FakeSpineDataset(num_samples=300, img_size=64)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    model = resnet18(in_channels=1, num_classes=3).to(device)
    print(f"ResNet-18 params: {count_parameters(model):,}")

    weights = load_class_weights("data/train.csv")
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(5):
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        print(f"Epoch {epoch+1}/5 - loss: {total_loss/total:.4f} - acc: {correct/total:.2%}")

    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, 64, 64).to(device)
        out = model(x)
        pred = out.argmax(1).item()
        print(f"Demo predict 1 anh 64x64 -> lop {pred} ({CLASS_NAMES[pred]})")


if __name__ == "__main__":
    main()
