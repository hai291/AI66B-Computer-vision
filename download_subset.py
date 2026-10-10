"""Tai subset anh DICOM THAT cho issue #4: chi cac slice co annotation.

Doc data/train_label_coordinates.csv -> lay N studies dau tien trong train.csv
-> tai moi (study, series, instance) duy nhat ve data/train_images/.
Moi annotation sau nay = 1 mau train (crop quanh x, y).
"""

import os
import shutil

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "rsna-2024-lumbar-spine-degenerative-classification"
N_STUDIES = 12
OUT_ROOT = "data/train_images"


def main():
    coords = pd.read_csv("data/train_label_coordinates.csv")
    studies = pd.read_csv("data/train.csv")["study_id"].tolist()[:N_STUDIES]
    slices = (coords[coords.study_id.isin(studies)]
              [["study_id", "series_id", "instance_number"]]
              .drop_duplicates()
              .reset_index(drop=True))
    print(f"{len(studies)} studies -> {len(slices)} slices can tai")

    api = KaggleApi()
    api.authenticate()

    tmp = "_dl_tmp"
    os.makedirs(tmp, exist_ok=True)
    ok, miss, skip = 0, 0, 0
    for i, r in slices.iterrows():
        study, series, inst = int(r.study_id), int(r.series_id), int(r.instance_number)
        out = os.path.join(OUT_ROOT, str(study), str(series), f"{inst}.dcm")
        if os.path.exists(out):
            skip += 1
            continue
        try:
            api.competition_download_file(
                COMPETITION, f"train_images/{study}/{series}/{inst}.dcm",
                path=tmp, force=True, quiet=True)
            dl = os.path.join(tmp, f"{inst}.dcm")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.move(dl, out)
            ok += 1
        except Exception as e:
            miss += 1
            print(f"MISS {study}/{series}/{inst}: {e}")
        if (ok + miss + skip) % 20 == 0:
            print(f"... {ok + miss + skip}/{len(slices)} (ok={ok}, miss={miss}, skip={skip})")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"XONG: ok={ok}, miss={miss}, skip={skip}")


if __name__ == "__main__":
    main()
