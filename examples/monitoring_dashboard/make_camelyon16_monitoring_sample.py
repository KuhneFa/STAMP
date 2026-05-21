from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "torchmil/Camelyon16_MIL"
LOCAL_DATA_DIR = Path(__file__).parent / "data" / "camelyon16"
OUTPUT_CSV = Path(__file__).parent / "camelyon16_monitoring_predictions.csv"


def download_metadata() -> tuple[Path, Path]:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits_path = Path(
        hf_hub_download(
            REPO_ID,
            filename="dataset/splits.csv",
            repo_type="dataset",
            local_dir=LOCAL_DATA_DIR,
        )
    )
    labels_path = Path(
        hf_hub_download(
            REPO_ID,
            filename="dataset/patches_512/labels.tar.gz",
            repo_type="dataset",
            local_dir=LOCAL_DATA_DIR,
        )
    )
    return splits_path, labels_path


def load_labels(labels_path: Path) -> dict[str, int]:
    labels = {}
    with tarfile.open(labels_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.name.endswith(".npy"):
                continue
            file_obj = archive.extractfile(member)
            if file_obj is None:
                continue
            label = int(np.load(io.BytesIO(file_obj.read())).reshape(-1)[0])
            labels[Path(member.name).stem] = label
    return labels


def simulate_prediction_score(label: int, split: str, rng: np.random.Generator) -> float:
    if label == 1:
        score = rng.beta(8, 3)
    else:
        score = rng.beta(3, 8)

    if split == "test":
        # Simulate deployment drift: external/test cases become less decisive.
        score = 0.5 + (score - 0.5) * 0.65 + rng.normal(0.04, 0.05)
    else:
        score = score + rng.normal(0.0, 0.035)

    return float(np.clip(score, 0.01, 0.99))


def build_monitoring_table(splits_path: Path, labels_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    splits = pd.read_csv(splits_path)
    labels = load_labels(labels_path)

    rows = []
    for index, row in splits.sort_values("bag_name").reset_index(drop=True).iterrows():
        slide_id = str(row["bag_name"])
        split = str(row["split"])
        label = labels[slide_id]
        prediction_score = simulate_prediction_score(label, split, rng)
        center = (
            "Radboud" if slide_id.startswith("tumor") else "Utrecht"
        ) if split == "train" else "External"
        scanner = "scanner_A" if index % 3 == 0 else "scanner_B" if index % 3 == 1 else "scanner_C"
        true_label = "tumor" if label == 1 else "normal"
        predicted_label = "tumor" if prediction_score >= 0.5 else "normal"

        rows.append(
            {
                "patient_id": slide_id,
                "slide_id": slide_id,
                "cohort": "internal" if split == "train" else "external",
                "center": center,
                "scanner": scanner,
                "date": (pd.Timestamp("2025-01-01") + pd.Timedelta(days=index)).date(),
                "prediction_score": round(prediction_score, 4),
                "predicted_label": predicted_label,
                "true_label": true_label,
                "source": "CAMELYON16_MIL labels; prediction scores simulated",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    splits_path, labels_path = download_metadata()
    table = build_monitoring_table(splits_path, labels_path)
    table.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(table)} rows to {OUTPUT_CSV}")
    print(table["true_label"].value_counts().to_string())
    print(table["cohort"].value_counts().to_string())


if __name__ == "__main__":
    main()
