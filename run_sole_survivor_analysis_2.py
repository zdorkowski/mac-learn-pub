from __future__ import annotations

import os
import warnings
import pandas as pd

from sole_survivor_utils_2 import (
    # Data prep + feature utilities
    standardize_columns,
    detect_feature_types,
    correlations_with_target,
    predictor_independence_matrix,
    find_high_corr_pairs,

    # Visual diagnostics
    plot_corr_heatmap,
    plot_pred_vs_actual,
    plot_residuals,

    # Modeling I/O
    build_design_matrices,
    fit_evaluate_linear_regression,
    align_and_predict_next,

    # make_feature_selector,  # (optional) uncomment if you want selection here
)

warnings.filterwarnings("ignore", message="X has feature names")

# Configure input/output file paths (adjust if needed)
PAST_CSV = "sole_survivor_past.csv"
NEXT_CSV = "sole_survivor_next.csv"

# Candidate names for the target column in the past CSV.
TARGET_CANDIDATES = ["survivalscore"]

# Where to write the predictions CSV
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def banner(title: str) -> None:
    # Pretty console header to separate major phases so logs are easy to skim.
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def detect_target_column(df: pd.DataFrame) -> str:
    # Determine which column should be used as the target (y).
    # Tries each name in TARGET_CANDIDATES in order; raises error if none are found.
    cols = df.columns.tolist()  # ensure we actually call .tolist()
    for candidate in TARGET_CANDIDATES:
        if candidate in cols:
            return candidate
    raise KeyError(
        f"Could not find any target column in {TARGET_CANDIDATES}. Found columns: {cols}"
    )


# -------------------------------------------------------------------
# Minimal, modular helpers
# -------------------------------------------------------------------

def load_and_prepare(past_csv: str, next_csv: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    # Load both CSVs, standardize their column names, detect the target column,
    # coerce the target to numeric, and drop rows that are missing a target.
    # Returns:
    #   df_past: cleaned historical dataset with non-null target
    #   df_next: cleaned next-season dataset (no target required)
    #   target:  the name of the target column (string)
    banner("1) Load Data")
    df_past = pd.read_csv(past_csv)
    df_next = pd.read_csv(next_csv)
    print(f"Loaded past: {df_past.shape[0]:,} rows, {df_past.shape[1]} cols")
    print(f"Loaded next: {df_next.shape[0]:,} rows, {df_next.shape[1]} cols")

    # Normalize headers (lowercase, remove spaces/dashes) so names match utils
    df_past = standardize_columns(df_past)
    df_next = standardize_columns(df_next)

    # Pick the correct target column name based on what exists
    target = detect_target_column(df_past)
    print(f"Using target column: {target}")

    # Ensure target is numeric, then drop rows lacking a valid target score
    df_past[target] = pd.to_numeric(df_past[target], errors="coerce")
    before = len(df_past)
    df_past = df_past.dropna(subset=[target])
    print(f"Dropped {before - len(df_past)} rows with missing target; remaining {len(df_past)}.")

    return df_past, df_next, target


def run_diagnostics(df_past: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    # Print quick dataset diagnostics:
    #   - Which features are numeric vs categorical
    #   - Top numeric feature ↔ target correlations
    #   - Predictor–predictor correlation matrix and highly correlated pairs
    #   - A heatmap of numeric predictor correlations (upper triangle)
    # Returns the detected numeric and categorical feature name lists for reuse.
    banner("2) Feature Detection & Diagnostics")
    numeric_feats, categorical_feats = detect_feature_types(df_past, target)
    print(f"Numeric features ({len(numeric_feats)}): {numeric_feats}")
    print(f"Categorical features ({len(categorical_feats)}): {categorical_feats}")

    # Feature ↔ target correlations (numeric-only)
    corr_target = correlations_with_target(df_past, target, numeric_feats)
    if not corr_target.empty:
        print("\nTop feature ↔ target correlations (by |r|):")
        print(corr_target.head(10).to_string())
    else:
        print("\nNo numeric features available to compute correlations with target.")

    # Predictor–predictor correlation, useful to spot multicollinearity
    corr_pred = predictor_independence_matrix(df_past, numeric_feats)
    high_pairs = find_high_corr_pairs(corr_pred, threshold=0.8)
    if high_pairs:
        print("\nHighly correlated predictor pairs (|r| > 0.80):")
        for a, b, v in high_pairs:
            print(f" - {a} ↔ {b} (r = {v:.2f})")
    else:
        print("\nNo highly correlated numeric predictor pairs above |r| > 0.80.")

    # Visualize correlation structure across numeric predictors
    plot_corr_heatmap(corr_pred, "Predictor–Predictor Correlation (Independence)")
    return numeric_feats, categorical_feats


def train_and_plot(
    df_past: pd.DataFrame,
    target: str,
    numeric_feats: list[str],
    categorical_feats: list[str],
    *,
    test_size: float = 0.25,
    random_state: int = 99,
    selector=None,  # Optional: pass a feature selector object from utils if desired
):
    # Build the model-ready matrices (X, y), optionally apply feature selection
    # (fit on TRAIN only inside the utils), fit LinearRegression, print train/test
    # metrics, and render basic plots (Pred vs Actual for train/test + residuals).
    #
    # Returns:
    #   artifacts:         the trained model + predictions/metrics/selector info
    #   numeric_feats:     (possibly unchanged) list of numeric feature names
    #   categorical_feats: list of categorical feature names actually used
    banner("3) Modeling — Linear Regression (Train/Test)")

    # One-hot encode categoricals (dropping ID-like columns inside utils) and build y
    X, y, numeric_feats, categorical_feats = build_design_matrices(
        df=df_past,
        target=target,
        numeric_feats=numeric_feats,
        categorical_feats=categorical_feats
    )

    # Train/eval the model (and feature selector if provided)
    artifacts = fit_evaluate_linear_regression(
        X, y, test_size=test_size, random_state=random_state, selector=selector
    )

    # Console metrics for quick evaluation
    print(f"Train R^2:  {artifacts.train_metrics['r2']:.3f}")
    print(f"Train RMSE: {artifacts.train_metrics['rmse']:.3f}")
    print(f"Test  R^2:  {artifacts.test_metrics['r2']:.3f}")
    print(f"Test  RMSE: {artifacts.test_metrics['rmse']:.3f}")

    # Plots to inspect fit quality and residual behavior
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

    return artifacts, numeric_feats, categorical_feats


def predict_and_save(
    artifacts,
    df_next: pd.DataFrame,
    numeric_feats: list[str],
    categorical_feats: list[str],
    output_dir: str,
) -> None:
    # Use the trained model + (optional) selector to predict next-season scores:
    #   - Build next-season design matrix that matches training columns
    #   - Apply the same selector mask/order if one was used
    #   - Write a full CSV of predictions to disk
    #   - Print a top-3 table in descending predicted score
    banner("4) Next Season — Predictions")

    # Create next-season X with same column layout; predict scores
    preds_next = align_and_predict_next(
        model=artifacts.model,
        train_X_cols=artifacts.feature_columns,
        next_df=df_next,
        numeric_feats=numeric_feats,
        categorical_feats=categorical_feats,
        selector=artifacts.selector,  # safe even if None
    )

    # Defensive check to avoid NoneType errors if something upstream changed
    if preds_next is None or "predicted_survivalscore" not in preds_next.columns:
        raise RuntimeError("align_and_predict_next did not return a valid DataFrame with 'predicted_survivalscore'.")

    # Rank contestants by predicted score
    preds_next_sorted = preds_next.sort_values("predicted_survivalscore", ascending=False).reset_index(drop=True)

    # Persist complete predictions for review/hand-off
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "next_season_predictions.csv")
    preds_next_sorted.to_csv(out_path, index=False)
    print(f"Saved full predictions: {out_path}")

    # Console summary: top 3 candidates by predicted score
    print("\nTop 3 predicted performers (descending by predicted_survivalscore):")
    display_cols = [c for c in preds_next_sorted.columns if c != "predicted_survivalscore"]
    top3 = preds_next_sorted[["predicted_survivalscore"] + display_cols].head(3)
    print(top3.to_string(index=False))


# -------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------

def main() -> None:
    # Orchestrate the full workflow:
    #   1) Load & clean data; confirm target
    #   2) Run quick diagnostics (correlations/heatmap)
    #   3) Train/test the linear regression (optional selection)
    #   4) Predict next-season scores and save rankings
    df_past, df_next, target = load_and_prepare(PAST_CSV, NEXT_CSV)

    # Brief “X-ray” of the dataset to sanity check relationships
    numeric_feats, categorical_feats = run_diagnostics(df_past, target)

    # (Optional) add a feature selector from utils, e.g.:
    # selector = make_feature_selector(method="lasso_sfmodel")
    selector = None

    # Train/evaluate and produce plots
    artifacts, numeric_feats, categorical_feats = train_and_plot(
        df_past, target, numeric_feats, categorical_feats,
        test_size=0.25, random_state=99, selector=selector
    )

    # Predict next season and write outputs
    predict_and_save(artifacts, df_next, numeric_feats, categorical_feats, OUTPUT_DIR)


if __name__ == "__main__":
    main()