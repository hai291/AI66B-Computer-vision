import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import LumbarCNN  # lay CNN tu model.py


class FakeSpineDataset(Dataset):
    # Gia lap crop MRI + nhan 0/1/2 (Nhe/Vua/Nang)
    def __init__(self, num_samples=300, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Bia anh
        label = torch.randint(0, 3, (1,)).item()  # chon nhau nhien 0/1/2
        img = torch.randn(1, self.img_size, self.img_size)
        c = self.img_size // 2
        if label == 0:  # Normal/Mild: lam toi vung giua
            img[:, c-4:c+4, c-4:c+4] -= 0.7
        elif label == 2:  # Severe: lam sang vung giua
            img[:, c-4:c+4, c-4:c+4] += 0.7
        # label 1 Moderate: de nguyen
        return img, label


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = FakeSpineDataset(num_samples=300, img_size=64)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # chia mam 16 anh

    model = LumbarCNN(in_channels=1, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()  # do doan sai bao nhieu
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # tu sua trong so model

    model.train()
    for epoch in range(5):
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()  # xoa gradient cu
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()  # tinh huong sua sai
            optimizer.step()  # cap nhat model

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        print(f"Epoch {epoch+1}/5 - loss: {total_loss/total:.4f} - acc: {correct/total:.2%}")

    # Thi thu 1 anh
    model.eval()
    with torch.no_grad():  # thi thi khong can tinh gradient
        x = torch.randn(1, 1, 64, 64).to(device)
        out = model(x)
        pred = out.argmax(1).item()
        print(f"Demo predict 1 anh 64x64 -> logits {out.cpu().tolist()} -> lop {pred} "
              f"(0=Normal/Mild, 1=Moderate, 2=Severe)")


if __name__ == "__main__":
    main()
