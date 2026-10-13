import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss


def summarize(root):
    names = ["baseline", "densenet_manual", "densenet"]
    titles = ["CNN baseline", "DenseNet manual", "DenseNet thư viện"]
    labels = ["Normal/Mild", "Moderate", "Severe"]
    probability_columns = ["normal_mild", "moderate", "severe"]
    samples = pd.read_csv(root / names[0] / "samples.csv")
    split = pd.read_csv(root / names[0] / "split.csv")
    assert not split.study_id.duplicated().any(), "Một ca có nhiều dòng split"
    assert set(split.split) == {"train", "val", "test"}
    actual_split = samples[["study_id", "split"]].drop_duplicates()
    pd.testing.assert_frame_equal(
        actual_split.sort_values("study_id").reset_index(drop=True),
        split.sort_values("study_id").reset_index(drop=True),
    )
    key_columns = ["study_id", "series_id", "instance_number", "condition", "level"]
    assert not samples.duplicated(key_columns).any(), "Crop bị lặp"
    configs, rows, matrices = [], [], []
    for name in names:
        folder = root / name
        pd.testing.assert_frame_equal(samples, pd.read_csv(folder / "samples.csv"))
        pd.testing.assert_frame_equal(split, pd.read_csv(folder / "split.csv"))
        config = json.loads((folder / "config.json").read_text())
        assert config["model"] == name
        configs.append(config)
        history = pd.read_csv(folder / "history.csv")
        row = {"model": name, "pretrained": config["pretrained"]}
        for part in ["val", "test"]:
            predictions = pd.read_csv(folder / f"{part}_predictions.csv")
            expected = samples[samples.split == part].reset_index(drop=True)
            pd.testing.assert_frame_equal(expected, predictions[samples.columns])
            probabilities = predictions[probability_columns].to_numpy()
            assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
            assert (probabilities <= 1).all()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-7)
            targets = predictions.label.to_numpy()
            predicted = probabilities.argmax(axis=1)
            metrics = {
                "log_loss": log_loss(targets, probabilities, labels=[0, 1, 2]),
                "accuracy": accuracy_score(targets, predicted),
                "macro_f1": f1_score(targets, predicted, labels=[0, 1, 2],
                                     average="macro", zero_division=0),
            }
            matrix = confusion_matrix(targets, predicted, labels=[0, 1, 2])
            saved = json.loads((folder / f"{part}_metrics.json").read_text())
            assert saved["split"] == part and saved["crops"] == len(expected)
            assert saved["epoch"] == int(history.loc[history.val_loss.idxmin(), "epoch"])
            np.testing.assert_array_equal(matrix, saved["confusion_matrix"])
            np.testing.assert_array_equal(np.bincount(targets, minlength=3), saved["class_counts"])
            for metric, value in metrics.items():
                np.testing.assert_allclose(value, saved[metric], rtol=0, atol=1e-10)
                row[f"{part}_{metric}"] = value
            if part == "val":
                best = history[history.epoch == saved["epoch"]].iloc[0]
                np.testing.assert_allclose(
                    [best.val_loss, best.accuracy, best.macro_f1],
                    [metrics["log_loss"], metrics["accuracy"], metrics["macro_f1"]], atol=1e-10,
                )
            else:
                matrices.append(matrix)
                row["epoch"] = saved["epoch"]
        exported = pd.read_csv(folder / "comparison.csv").iloc[0]
        assert exported["model"] == name and exported["pretrained"] == config["pretrained"]
        for field in ["epoch", "test_log_loss", "test_accuracy", "test_macro_f1"]:
            np.testing.assert_allclose(exported[field], row[field], rtol=0, atol=1e-10)
        rows.append(row)

    for field in ["pipeline_version", "seed", "epochs", "max_studies", "crop_size",
                  "image_size", "batch_size", "lr", "pretrained"]:
        assert len({config[field] for config in configs}) == 1, f"Khác cấu hình: {field}"
    comparison = pd.DataFrame(rows)
    comparison.to_csv(root / "comparison.csv", index=False)
    counts = pd.crosstab(samples.split, samples.label).reindex(
        index=["train", "val", "test"], columns=[0, 1, 2], fill_value=0,
    )
    counts.columns = labels
    counts.insert(0, "studies", samples.groupby("split").study_id.nunique())
    counts.insert(1, "crops", samples.groupby("split").size())
    counts.to_csv(root / "split_summary.csv")

    colors = ["#536878", "#247A68", "#AA6034"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, metric, title in zip(axes, ["log_loss", "accuracy", "macro_f1"],
                                ["Log loss (thấp hơn tốt hơn)", "Accuracy", "Macro F1"]):
        bars = ax.bar(titles, comparison[f"test_{metric}"], color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.set_title(title)
        ax.set_ylim(0, max(1, comparison[f"test_{metric}"].max() * 1.2))
        ax.tick_params(axis="x", labelsize=9, rotation=12)
        ax.spines[["top", "right"]].set_visible(False)
    test_count = counts.loc["test"]
    fig.suptitle(
        f"Test trên cùng {test_count['crops']} crop / {test_count['studies']} ca"
        f" — {len(split)} ca, {configs[0]['epochs']} epoch"
    )
    fig.savefig(root / "test_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    maximum = max(matrix.max() for matrix in matrices)
    for ax, title, matrix in zip(axes, titles, matrices):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        ax.set_xticks(range(3), labels, rotation=20, ha="right")
        ax.set_yticks(range(3), labels)
        ax.set_title(title)
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thật")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > maximum / 2 else "black")
    fig.suptitle("Confusion matrix trên test — cùng thang màu theo số crop")
    fig.savefig(root / "test_confusion_matrices.png", dpi=180)
    plt.close(fig)
    print("PASS: same samples/splits/configuration; validation/test metrics match predictions.")
    print(counts.to_string())
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra và vẽ kết quả đã lưu, không train lại.")
    parser.add_argument("root", nargs="?", type=Path, default=Path("results/issue5_smoke_v2"))
    summarize(parser.parse_args().root)
