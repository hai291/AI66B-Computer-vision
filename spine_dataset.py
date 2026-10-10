"""Dataset ANH THAT cho issue #4: DICOM -> crop quanh (x, y) -> nhan tu train.csv.

Moi dong trong train_label_coordinates.csv = 1 mau train:
  anh = crop CROP_SIZE x CROP_SIZE quanh diem bac si danh dau tren slice do
  nhan = o train.csv tai cot <condition>_<level> (0=Mild, 1=Moderate, 2=Severe)
Bo cac annotation khong co nhan (NaN) hoac thieu file DICOM.
"""

import os

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

LABEL2IDX = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
CROP_SIZE = 128


def coord_to_column(condition, level):
    # "Spinal Canal Stenosis" + "L1/L2" -> "spinal_canal_stenosis_l1_l2"
    return condition.lower().replace(" ", "_") + "_" + level.lower().replace("/", "_")


def build_samples(img_root="data/train_images",
                  train_csv="data/train.csv",
                  coords_csv="data/train_label_coordinates.csv"):
    """Quet anh da tai ve, ghep voi 2 CSV -> DataFrame moi hang 1 mau train."""
    labels = pd.read_csv(train_csv).set_index("study_id")
    coords = pd.read_csv(coords_csv)

    # Chi giu annotation cua study da tai anh
    downloaded = {int(d) for d in os.listdir(img_root)
                  if os.path.isdir(os.path.join(img_root, d))}
    coords = coords[coords.study_id.isin(downloaded)]

    rows = []
    for r in coords.itertuples():
        col = coord_to_column(r.condition, r.level)
        if col not in labels.columns:
            continue
        sev = labels.at[r.study_id, col]
        if not isinstance(sev, str) or sev not in LABEL2IDX:  # nhan thieu (NaN)
            continue
        path = os.path.join(img_root, str(r.study_id),
                            str(r.series_id), f"{int(r.instance_number)}.dcm")
        if not os.path.exists(path):
            continue
        rows.append({"path": path, "x": float(r.x), "y": float(r.y),
                     "label": LABEL2IDX[sev], "study_id": r.study_id})
    df = pd.DataFrame(rows)
    print(f"mau that: {len(df)} crops tu {df.study_id.nunique()} studies "
          f"(bo {len(coords) - len(df)} annotation thieu nhan/anh)")
    return df


def load_crop(path, x, y, size=CROP_SIZE):
    """Doc DICOM -> cat vung size x size quanh (x, y) -> chuan hoa [0,1]."""
    img = pydicom.dcmread(path).pixel_array.astype(np.float32)
    h, w = img.shape
    # Kep tam crop de khong tran vien
    cx = min(max(int(round(x)), size // 2), w - size // 2)
    cy = min(max(int(round(y)), size // 2), h - size // 2)
    crop = img[cy - size // 2: cy + size // 2, cx - size // 2: cx + size // 2]
    crop = (crop - crop.min()) / (crop.max() - crop.min() + 1e-6)
    return torch.from_numpy(crop).unsqueeze(0)  # 1 x H x W


class SpineCropDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        return load_crop(r["path"], r["x"], r["y"]), r["label"]
