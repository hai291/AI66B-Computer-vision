import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRAIN_SCRIPT = Path(__file__).with_name("train_resnet.py")

import numpy as np
import pandas as pd
import torch
from torch import nn
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from resnet.model_resnet import build_resnet
from resnet.model_resnet_manual import BasicBlock, build_resnet_manual
from resnet.train_resnet import read_samples


def make_fake_data(root):
    labels, coordinates = [], []
    for study in range(20):
        severity = ["Normal/Mild", "Moderate", "Severe"][study % 3]
        labels.append({"study_id": study, "spinal_canal_stenosis_l1_l2": severity})
        path = root / "train_images" / str(study) / "1" / "1.dcm"
        path.parent.mkdir(parents=True)
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        dicom = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        dicom.Rows, dicom.Columns, dicom.SamplesPerPixel = 48, 64, 1
        dicom.BitsAllocated, dicom.BitsStored, dicom.HighBit = 16, 16, 15
        dicom.PixelRepresentation = 0
        dicom.PhotometricInterpretation = "MONOCHROME2"
        dicom.PixelData = (np.arange(48 * 64, dtype=np.uint16) + study).tobytes()
        dicom.save_as(path, enforce_file_format=True)
        # Hai crop cua cung mot ca phai luon o cung mot tap.
        for x, y in [(0, 0), (63, 47)]:
            coordinates.append({"study_id": study, "series_id": 1, "instance_number": 1,
                                "condition": "Spinal Canal Stenosis", "level": "L1/L2", "x": x, "y": y})
    pd.DataFrame(labels).to_csv(root / "train.csv", index=False)
    pd.DataFrame(coordinates).to_csv(root / "train_label_coordinates.csv", index=False)


def check_architecture():
    # Skip connection doi shape: stride=2 vua giam khong gian vua tang kenh.
    block = BasicBlock(8, 16, stride=2)
    image = torch.randn(2, 8, 16, 16, requires_grad=True)
    output = block(image)
    assert output.shape == (2, 16, 8, 8)
    output.sum().backward()
    assert image.grad is not None
    assert all(p.grad is not None for p in block.parameters())

    # Identity shortcut: cung shape, cong truc tiep.
    identity = BasicBlock(16, 16, stride=1)
    assert len(identity.shortcut) == 0
    out = identity(torch.randn(2, 16, 8, 8))
    assert out.shape == (2, 16, 8, 8)

    manual, library = build_resnet_manual().eval(), build_resnet(False).eval()
    n_manual = sum(p.numel() for p in manual.parameters())
    n_library = sum(p.numel() for p in library.parameters())
    assert n_manual == n_library, f"manual {n_manual} != library {n_library}"
    kinds = (nn.Conv2d, nn.BatchNorm2d, nn.Linear)
    manual_layers = [m for m in manual.modules() if isinstance(m, kinds)]
    library_layers = [m for m in library.modules() if isinstance(m, kinds)]
    assert len(manual_layers) == len(library_layers), (
        f"manual layers {len(manual_layers)} != library {len(library_layers)}")
    # Chi trong test: gan cung trong so de kiem tra hai cach dung cung phep tinh.
    for own, reference in zip(manual_layers, library_layers):
        assert type(own) is type(reference)
        own.load_state_dict(reference.state_dict())
    with torch.no_grad():
        image = torch.randn(2, 1, 64, 64)
        torch.testing.assert_close(manual(image), library(image), rtol=1e-5, atol=1e-6)
    print(f"PASS: residual connections, gradients, manual/library ResNet18 numerical agreement ({n_manual:,} params)")


def check_training(root):
    split_file = root / "split.csv"
    samples = read_samples(root, 42, 0, split_file)
    assert samples.groupby("study_id").split.nunique().eq(1).all()
    assert samples.groupby("split").study_id.nunique().to_dict() == {"test": 3, "train": 14, "val": 3}
    pd.testing.assert_frame_equal(samples, read_samples(root, 99, 0, split_file))
    try:
        read_samples(root, 42, 8, split_file)
        raise AssertionError("Different subset should not reuse the same split file")
    except ValueError:
        pass

    # Lam hong anh test: training van phai chay duoc vi chua duoc doc tap test.
    test_images = {}
    for relative in samples.loc[samples.split == "test", "path"].unique():
        path = root / relative
        test_images[path] = path.read_bytes()
        path.write_bytes(b"test must not be read during training")
    names = ["baseline", "resnet_manual", "resnet"]
    common = [sys.executable, str(TRAIN_SCRIPT), "--data-root", str(root), "--workers", "0", "--batch-size", "4"]
    for name in names:
        output = root / name
        subprocess.run(common + ["--model", name, "--epochs", "1", "--output", str(output),
                                 "--split-file", str(split_file), "--image-size", "64", "--crop-size", "16"], check=True)
        assert not (output / "test_metrics.json").exists()
        original_val = json.loads((output / "val_metrics.json").read_text())
        subprocess.run(common + ["--evaluate", str(output), "--split", "val"], check=True)
        assert original_val == json.loads((output / "val_metrics.json").read_text())
        pd.testing.assert_frame_equal(pd.read_csv(output / "samples.csv"), samples, check_dtype=False)
    for path, content in test_images.items():
        path.write_bytes(content)
    for name in names:
        output = root / name
        subprocess.run(common + ["--evaluate", str(output)], check=True)
        metrics = json.loads((output / "test_metrics.json").read_text())
        predictions = pd.read_csv(output / "test_predictions.csv")
        assert metrics["split"] == "test" and metrics["crops"] == 6
        assert set(predictions.study_id) == set(samples.loc[samples.split == "test", "study_id"])
        assert np.isfinite(metrics["log_loss"])
    print("PASS: shared 70/15/15 split, no test reads during training, saved checkpoints, held-out test evaluation")


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_architecture()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        make_fake_data(root)
        check_training(root)
