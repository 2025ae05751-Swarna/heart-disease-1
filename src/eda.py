import json

import matplotlib.pyplot as plt
import seaborn as sns

from .config import METRICS_DIR, NUMERIC_FEATURES, PLOTS_DIR, TARGET
from .data import clean_data, load_raw_data


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    data_frame = clean_data(load_raw_data())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, feature in enumerate(NUMERIC_FEATURES):
        sns.histplot(data_frame[feature], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution: {feature}")
    axes[-1].axis("off")
    fig.suptitle("Clinical Feature Distribution Review")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_histograms.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(data_frame.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Overview")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    counts = data_frame[TARGET].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    ax.set_title("Outcome Balance")
    ax.set_xlabel("Diagnosis flag (0 = no disease, 1 = disease)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_balance.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=data_frame[NUMERIC_FEATURES], orient="h", ax=ax)
    ax.set_title("Numeric Feature Spread")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "boxplots_numeric.png", dpi=200)
    plt.close(fig)

    summary = {
        "row_count": len(data_frame),
        "features_plus_target": int(data_frame.shape[1]),
        "missing_values_by_column": data_frame.isna().sum().to_dict(),
        "class_counts": {str(key): int(value) for key, value in data_frame[TARGET].value_counts().sort_index().items()},
        "descriptive_statistics": data_frame.describe().round(3).to_dict(),
    }
    (METRICS_DIR / "data_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    main()
