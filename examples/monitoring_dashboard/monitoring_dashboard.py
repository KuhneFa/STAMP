from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

try:
    from scipy.stats import ks_2samp
except ImportError:
    ks_2samp = None


REQUIRED_COLUMNS = {"patient_id", "prediction_score"}
OPTIONAL_COLUMNS = {
    "slide_id",
    "cohort",
    "center",
    "scanner",
    "date",
    "predicted_label",
    "true_label",
}
IGNORED_METADATA_COLUMNS = {"source", "loss"}
STAMP_PATIENT_ALIASES = {"PATIENT": "patient_id", "patient": "patient_id"}
STAMP_PRED_ALIASES = {"pred": "predicted_label"}


@dataclass
class DriftResult:
    group: str
    n: int
    mean_score: float
    mean_uncertainty: float
    mean_difference: float
    ks_pvalue: float | None
    warning: bool


def load_predictions(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        sample_path = Path(__file__).with_name("sample_predictions.csv")
        df = pd.read_csv(sample_path)
    else:
        df = pd.read_csv(uploaded_file)

    df = normalize_columns(df)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s): {missing_list}")

    df["prediction_score"] = pd.to_numeric(df["prediction_score"], errors="coerce")
    df = df.dropna(subset=["patient_id", "prediction_score"]).copy()
    df["prediction_score"] = df["prediction_score"].clip(0.0, 1.0)

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return add_uncertainty(df)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={**STAMP_PATIENT_ALIASES, **STAMP_PRED_ALIASES}).copy()

    if "prediction_score" not in df.columns:
        probability_columns = infer_probability_columns(df)
        if probability_columns:
            df["prediction_score"] = df[probability_columns].max(axis=1)

    if "true_label" not in df.columns:
        label_candidates = [
            column
            for column in df.columns
            if column not in REQUIRED_COLUMNS
            and column not in OPTIONAL_COLUMNS
            and column not in IGNORED_METADATA_COLUMNS
            and not column.endswith("_mutated")
            and not column.endswith("_wild type")
            and not column.startswith("prob_")
        ]
        if label_candidates:
            df = df.rename(columns={label_candidates[0]: "true_label"})

    return df


def infer_probability_columns(df: pd.DataFrame) -> list[str]:
    ignored = {
        "patient_id",
        "slide_id",
        "cohort",
        "center",
        "scanner",
        "date",
        "predicted_label",
        "true_label",
        "loss",
    }
    candidates = []
    for column in df.columns:
        if column in ignored:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any() and values.between(0, 1).all():
            candidates.append(column)
    return candidates


def add_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["uncertainty"] = 1 - (df["prediction_score"] - 0.5).abs() * 2
    df["uncertainty"] = df["uncertainty"].clip(0.0, 1.0)
    df["confidence"] = 1 - df["uncertainty"]
    return df


def dataset_stats(df: pd.DataFrame) -> dict[str, object]:
    return {
        "patients": df["patient_id"].nunique(),
        "slides": df["slide_id"].nunique() if df["slide_id"].notna().any() else "n/a",
        "cohorts": sorted(df["cohort"].dropna().unique().tolist()),
        "centers": sorted(df["center"].dropna().unique().tolist()),
        "mean_score": df["prediction_score"].mean(),
    }


def make_histogram(
    values: pd.Series, title: str, xlabel: str, threshold: float | None = None
):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values.dropna(), bins=20, color="#4c78a8", edgecolor="white")
    if threshold is not None:
        ax.axvline(threshold, color="#d62728", linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cases")
    ax.grid(alpha=0.2)
    return fig


def make_group_boxplot(df: pd.DataFrame, group_column: str):
    grouped = [
        group["prediction_score"].dropna().to_numpy()
        for _, group in df.dropna(subset=[group_column]).groupby(group_column)
    ]
    labels = [
        str(name) for name, _ in df.dropna(subset=[group_column]).groupby(group_column)
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(grouped, labels=labels, vert=True)
    ax.set_title(f"Prediction score distribution by {group_column}")
    ax.set_xlabel(group_column)
    ax.set_ylabel("Prediction score")
    ax.grid(alpha=0.2)
    return fig


def compute_drift(
    df: pd.DataFrame,
    group_column: str,
    reference_group: str,
    mean_shift_threshold: float,
    pvalue_threshold: float,
) -> list[DriftResult]:
    grouped = df.dropna(subset=[group_column]).groupby(group_column)
    if reference_group not in grouped.groups:
        return []

    reference_scores = grouped.get_group(reference_group)["prediction_score"].dropna()
    reference_mean = reference_scores.mean()
    results = []

    for group_name, group_df in grouped:
        scores = group_df["prediction_score"].dropna()
        if scores.empty:
            continue
        pvalue = None
        if (
            ks_2samp is not None
            and group_name != reference_group
            and len(scores) >= 2
            and len(reference_scores) >= 2
        ):
            pvalue = float(ks_2samp(reference_scores, scores).pvalue)
        mean_difference = float(scores.mean() - reference_mean)
        warning = abs(mean_difference) >= mean_shift_threshold or (
            pvalue is not None and pvalue < pvalue_threshold
        )
        results.append(
            DriftResult(
                group=str(group_name),
                n=int(len(scores)),
                mean_score=float(scores.mean()),
                mean_uncertainty=float(group_df["uncertainty"].mean()),
                mean_difference=mean_difference,
                ks_pvalue=pvalue,
                warning=warning,
            )
        )

    return results


def has_labels(df: pd.DataFrame) -> bool:
    return "true_label" in df.columns and df["true_label"].notna().any()


def compute_performance(df: pd.DataFrame) -> tuple[dict[str, float], np.ndarray, list[str]]:
    labeled = df.dropna(subset=["true_label", "predicted_label"]).copy()
    if labeled.empty:
        return {}, np.array([]), []

    y_true = labeled["true_label"].astype(str)
    y_pred = labeled["predicted_label"].astype(str)
    labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}

    if y_true.nunique() == 2:
        positive_label = sorted(y_true.unique())[-1]
        y_binary = (y_true == positive_label).astype(int)
        if y_binary.nunique() == 2:
            metrics["auroc"] = float(roc_auc_score(y_binary, labeled["prediction_score"]))

    return metrics, confusion_matrix(y_true, y_pred, labels=labels), labels


def make_confusion_matrix(matrix: np.ndarray, labels: list[str]):
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    return fig


def make_calibration_plot(df: pd.DataFrame):
    labeled = df.dropna(subset=["true_label"])
    if labeled["true_label"].nunique() != 2 or len(labeled) < 10:
        return None
    positive_label = sorted(labeled["true_label"].astype(str).unique())[-1]
    y_true = (labeled["true_label"].astype(str) == positive_label).astype(int)
    prob_true, prob_pred = calibration_curve(
        y_true, labeled["prediction_score"], n_bins=min(10, len(labeled))
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, marker="o", label="Observed")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Fraction positive")
    ax.set_title("Calibration curve")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def main() -> None:
    st.set_page_config(page_title="STAMP Monitoring Dashboard", layout="wide")

    st.title("STAMP Monitoring & Uncertainty Dashboard")
    st.write(
        "Exploratory reliability prototype for STAMP-like prediction outputs. "
        "This dashboard is not clinically validated and should not be used for medical decisions."
    )

    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader("Prediction CSV", type=["csv"])
        uncertainty_threshold = st.slider(
            "Review threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05
        )
        mean_shift_threshold = st.slider(
            "Drift mean-shift warning", min_value=0.0, max_value=0.5, value=0.15, step=0.01
        )
        pvalue_threshold = st.slider(
            "KS p-value warning", min_value=0.001, max_value=0.2, value=0.05, step=0.001
        )

    try:
        df = load_predictions(uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    df["review_flag"] = df["uncertainty"] >= uncertainty_threshold
    stats = dataset_stats(df)

    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients", stats["patients"])
    col2.metric("Slides", stats["slides"])
    col3.metric("Cohorts", len(stats["cohorts"]))
    col4.metric("Mean score", f"{stats['mean_score']:.2f}")

    with st.expander("Preview normalized data"):
        st.dataframe(df.head(50), use_container_width=True)

    st.header("Prediction And Uncertainty")
    col1, col2 = st.columns(2)
    col1.pyplot(make_histogram(df["prediction_score"], "Prediction scores", "Score"))
    col2.pyplot(
        make_histogram(
            df["uncertainty"],
            "Uncertainty",
            "Uncertainty",
            threshold=uncertainty_threshold,
        )
    )
    st.info(
        "Uncertainty is a simple score-distance heuristic: highest near 0.5, lowest near 0 or 1."
    )

    st.header("Drift Analysis")
    group_options = [
        column
        for column in ["cohort", "center", "scanner"]
        if df[column].notna().any() and df[column].nunique() >= 2
    ]
    if df["date"].notna().any():
        df["month"] = df["date"].dt.to_period("M").astype(str)
        if df["month"].nunique() >= 2:
            group_options.append("month")

    if not group_options:
        st.warning("No cohort, center, scanner, or date grouping is available for drift analysis.")
    else:
        group_column = st.selectbox("Compare score distributions by", group_options)
        groups = sorted(df[group_column].dropna().astype(str).unique())
        reference_group = st.selectbox("Reference group", groups)
        drift_results = compute_drift(
            df,
            group_column,
            reference_group,
            mean_shift_threshold,
            pvalue_threshold,
        )
        drift_table = pd.DataFrame([result.__dict__ for result in drift_results])
        st.dataframe(drift_table, use_container_width=True)
        st.pyplot(make_group_boxplot(df, group_column))

        warnings = drift_table[drift_table["warning"]] if not drift_table.empty else pd.DataFrame()
        for _, row in warnings.iterrows():
            if row["group"] != reference_group:
                st.warning(
                    f"The score distribution for {group_column}={row['group']} differs from "
                    "the reference group. This may indicate dataset shift and should be reviewed."
                )

    st.header("Performance Monitoring")
    if not has_labels(df):
        st.warning(
            "Ground truth labels are not available. Performance monitoring is disabled, "
            "but uncertainty and drift monitoring are still possible."
        )
    elif df["predicted_label"].isna().all():
        st.warning("Ground truth labels exist, but predicted_label is missing.")
    else:
        metrics, matrix, labels = compute_performance(df)
        metric_cols = st.columns(max(1, len(metrics)))
        for idx, (name, value) in enumerate(metrics.items()):
            metric_cols[idx].metric(name.upper(), f"{value:.3f}")
        if matrix.size:
            st.pyplot(make_confusion_matrix(matrix, labels))
        calibration_fig = make_calibration_plot(df)
        if calibration_fig is None:
            st.info("Calibration curve requires at least 10 labeled binary cases.")
        else:
            st.pyplot(calibration_fig)

    st.header("Flagged Cases For Review")
    flagged = df[df["review_flag"]].sort_values("uncertainty", ascending=False)
    st.write(f"{len(flagged)} cases exceed the current review threshold.")
    st.dataframe(flagged, use_container_width=True)
    st.download_button(
        "Export flagged cases",
        data=flagged.to_csv(index=False),
        file_name="flagged_cases.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
