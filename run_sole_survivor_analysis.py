"""
run_sole_survivor_analysis.py

Console/plot-driven analysis (no markdown files).
- Loads CSVs with plain pd.read_csv
- Prints diagnostics, metrics, and top-3 predictions
- Shows heatmap, Pred vs Actual (train/test), and residual plot

USAGE:
    python run_sole_survivor_analysis.py

Files expected in the same folder:
    - sole_survivor_past.csv
    - sole_survivor_next.csv
"""

from __future__ import annotations

import os
import warnings
import pandas as pd

from sole_survivor_utils import (
    standardize_columns,
    detect_feature_types,
    correlations_with_target,
    predictor_independence_matrix,
    find_high_corr_pairs,
    plot_corr_heatmap,
    plot_pred_vs_actual,
    plot_residuals,
    build_design_matrices,
    fit_evaluate_linear_regression,
    align_and_predict_next,
)

warnings.filterwarnings("ignore", message="X has feature names")

# Adjust these filenames if needed
PAST_CSV = "sole_survivor_past.csv"
NEXT_CSV = "sole_survivor_next.csv"

# Try to detect target column name
TARGET_CANDIDATES = ["survivalscore"]

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def banner(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def detect_target_column(df: pd.DataFrame) -> str:
    """Return which target column to use."""
    cols = df.columns.tolist()
    for candidate in TARGET_CANDIDATES:
        if candidate in cols:
            return candidate
    raise KeyError(f"Could not find any target column in {TARGET_CANDIDATES}. Found columns: {cols}")


def main():
    # 1) Load data
    banner("1) Load Data")
    df_past = pd.read_csv(PAST_CSV)
    df_next = pd.read_csv(NEXT_CSV)
    print(f"Loaded past: {df_past.shape[0]:,} rows, {df_past.shape[1]} cols")
    print(f"Loaded next: {df_next.shape[0]:,} rows, {df_next.shape[1]} cols")

    # 2) Standardize schema & detect target
    df_past = standardize_columns(df_past)
    df_next = standardize_columns(df_next)

    target = detect_target_column(df_past)
    print(f"Using target column: {target}")

    df_past[target] = pd.to_numeric(df_past[target], errors="coerce")
    before = len(df_past)
    df_past = df_past.dropna(subset=[target])
    print(f"Dropped {before - len(df_past)} rows with missing target; remaining {len(df_past)}.")

    # 3) Feature detection & diagnostics
    banner("2) Feature Detection & Diagnostics")
    numeric_feats, categorical_feats = detect_feature_types(df_past, target)
    print(f"Numeric features ({len(numeric_feats)}): {numeric_feats}")
    print(f"Categorical features ({len(categorical_feats)}): {categorical_feats}")

    corr_target = correlations_with_target(df_past, target, numeric_feats)
    if not corr_target.empty:
        print("\nTop feature ↔ target correlations (by |r|):")
        print(corr_target.head(10).to_string())
    else:
        print("\nNo numeric features available to compute correlations with target.")

    corr_pred = predictor_independence_matrix(df_past, numeric_feats)
    high_pairs = find_high_corr_pairs(corr_pred, threshold=0.8)
    if high_pairs:
        print("\nHighly correlated predictor pairs (|r| > 0.80):")
        for a, b, v in high_pairs:
            print(f" - {a} ↔ {b} (r = {v:.2f})")
    else:
        print("\nNo highly correlated numeric predictor pairs above |r| > 0.80.")

    # Heatmap for independence
    plot_corr_heatmap(corr_pred, "Predictor–Predictor Correlation (Independence)")

    # 4) Modeling
    banner("3) Modeling — Linear Regression (Train/Test)")
    X, y, numeric_feats, categorical_feats = build_design_matrices(
        df=df_past,
        target=target,
        numeric_feats=numeric_feats,
        categorical_feats=categorical_feats
    )
    artifacts = fit_evaluate_linear_regression(X, y, test_size=0.25, random_state=99)

    print(f"Train R^2:  {artifacts.train_metrics['r2']:.3f}")
    print(f"Train RMSE: {artifacts.train_metrics['rmse']:.3f}")
    print(f"Test  R^2:  {artifacts.test_metrics['r2']:.3f}")
    print(f"Test  RMSE: {artifacts.test_metrics['rmse']:.3f}")

    # Plots
    plot_pred_vs_actual(
        artifacts.y_train, artifacts.y_train_pred,
        title="Linear Regression — TRAIN: Predicted vs Actual"
    )
    plot_pred_vs_actual(
        artifacts.y_test, artifacts.y_test_pred,
        title="Linear Regression — TEST: Predicted vs Actual"
    )
    plot_residuals(
        artifacts.y_test, artifacts.y_test_pred,
        title="Linear Regression — TEST: Residuals vs Predicted"
    )

    # 5) Predict next season
    banner("4) Next Season — Predictions")
    preds_next = align_and_predict_next(
        model=artifacts.model,
        train_X_cols=artifacts.feature_columns,
        next_df=df_next,
        numeric_feats=numeric_feats,
        categorical_feats=categorical_feats,
    )
    preds_next_sorted = preds_next.sort_values("predicted_survivalscore", ascending=False).reset_index(drop=True)

    # Save full predictions
    out_path = os.path.join(OUTPUT_DIR, "next_season_predictions.csv")
    preds_next_sorted.to_csv(out_path, index=False)
    print(f"Saved full predictions: {out_path}")

    # Print top 3
    print("\nTop 3 predicted performers (descending by predicted_survivalscore):")
    display_cols = [c for c in preds_next_sorted.columns if c != "predicted_survivalscore"]
    top3 = preds_next_sorted[["predicted_survivalscore"] + display_cols].head(3)
    print(top3.to_string(index=False))


if __name__ == "__main__":
    main()