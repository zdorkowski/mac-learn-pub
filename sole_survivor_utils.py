"""
sole_survivor_utils.py

Utilities for analyzing Sole Survivor data and training a regression model
to evaluate expert initial ratings and predict overall survival scores.

- Standardizes column names
- Auto-detects numeric vs categorical features
- Correlation diagnostics (linearity to target; independence among predictors)
- Linear Regression training & evaluation
- Plotting helpers (Pred vs Actual, Residuals, Corr Heatmap)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# -----------------------------
# Small helpers
# -----------------------------

# Standardizes DataFrame column names (lowercase, no spaces/dashes)
def standardize_columns(df):
    """Lowercase headers; remove spaces and dashes."""
    df = df.copy()
    df.columns = (
        df.columns.str.lower()
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    return df


# Calculates Root Mean Squared Error between true and predicted values
def rmse(y_true, y_pred):
    """Return the root mean squared error between actual and predicted values."""
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


# -----------------------------
# Feature diagnostics
# -----------------------------

# Identifies which columns are numeric vs categorical (ignores the target column)
def detect_feature_types(df, target):
    """Infer numeric and categorical feature lists automatically, excluding the target."""
    numeric_feats, categorical_feats = [], []
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_feats.append(c)
        elif (
            pd.api.types.is_bool_dtype(df[c])
            or pd.api.types.is_categorical_dtype(df[c])
            or df[c].dtype == object
        ):
            categorical_feats.append(c)
    return numeric_feats, categorical_feats


# Computes Pearson correlations between each numeric feature and the target
def correlations_with_target(df, target, features):
    """Pearson correlations of numeric features vs target."""
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return pd.Series(dtype=float)
    return (
        df[numeric + [target]]
        .corr()[target]
        .drop(target)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )


# Builds correlation matrix among numeric predictors to check for multicollinearity
def predictor_independence_matrix(df, features):
    """Predictor-predictor correlation matrix for numeric features."""
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return pd.DataFrame()
    return df[numeric].corr()


# Finds pairs of numeric predictors that are highly correlated (above threshold)
def find_high_corr_pairs(corr_mat, threshold=0.8):
    """Return highly correlated numeric predictor pairs above |threshold|."""
    pairs = []
    if corr_mat.empty:
        return pairs
    cols = corr_mat.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            val = float(corr_mat.loc[a, b])
            if abs(val) > threshold:
                pairs.append((a, b, val))
    return pairs


# -----------------------------
# Plotting helpers
# -----------------------------

# Draws a heatmap showing correlation between predictors
def plot_corr_heatmap(corr_mat, title):
    """Upper-triangular heatmap for predictor correlation matrix."""
    if corr_mat.empty:
        print("No numeric predictors available for independence heatmap.")
        return
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        corr_mat, mask=mask, annot=True, fmt=".2f",
        vmin=-1, vmax=1, center=0, cmap="coolwarm"
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Creates a scatter plot of predicted vs actual target values, with descriptive title
def plot_pred_vs_actual(y_true, y_pred, title, target_name="Survival Score"):
    """Predicted vs Actual scatter with 45° reference line and contextual title."""
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))

    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="w", linewidth=0.5)
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.title(title)
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.tight_layout()
    plt.show()


# Plots residuals (difference between actual and predicted) to check model fit
def plot_residuals(y_true, y_pred, title, target_name="Survival Score"):
    """Residuals vs Predicted to assess heteroscedasticity, with contextual title."""
    resid = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, resid, alpha=0.6, edgecolors="w", linewidth=0.5)
    plt.axhline(0, linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel(f"Predicted {target_name}")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Modeling pipeline
# -----------------------------

# Simple container for model, metrics, and predicted values
@dataclass
class ModelArtifacts:
    model: LinearRegression
    train_metrics: dict
    test_metrics: dict
    feature_columns: list
    y_train: np.ndarray
    y_train_pred: np.ndarray
    y_test: np.ndarray
    y_test_pred: np.ndarray


# Builds X (features) and y (target) with categorical encoding if necessary
def build_design_matrices(df, target, numeric_feats=None, categorical_feats=None):
    """Build X (with one-hot encoded categoricals) and y. Auto-detect features when None."""
    if numeric_feats is None or categorical_feats is None:
        auto_num, auto_cat = detect_feature_types(df, target)
        if numeric_feats is None:
            numeric_feats = auto_num
        if categorical_feats is None:
            categorical_feats = auto_cat

    # ⚠️ NEW CODE: drop identifier-like categorical features before encoding
    safe_cat_feats = [
        c for c in categorical_feats
        if c.lower() not in ("name", "id", "player", "contestant")
    ]

    # Optional: notify developer when identifiers are dropped
    dropped = set(categorical_feats) - set(safe_cat_feats)
    if dropped:
        print(f"⚠️ Dropped identifier-like categorical columns: {list(dropped)}")

    # Build data subset for modeling
    cols = [c for c in (numeric_feats + safe_cat_feats + [target]) if c in df.columns]
    data = df[cols].copy()

    X = pd.get_dummies(
        data[numeric_feats + safe_cat_feats],
        columns=safe_cat_feats, drop_first=True, dtype=int
    )
    y = data[target]
    return X, y, numeric_feats, safe_cat_feats


# Trains a linear regression model, evaluates it, and returns metrics + predictions
def fit_evaluate_linear_regression(X, y, test_size=0.25, random_state=99):
    """Train/test split, fit LinearRegression, compute metrics, and return results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    return ModelArtifacts(
        model=model,
        train_metrics={"r2": float(r2_score(y_train, y_pred_train)), "rmse": rmse(y_train, y_pred_train)},
        test_metrics={"r2": float(r2_score(y_test, y_pred_test)), "rmse": rmse(y_test, y_pred_test)},
        feature_columns=X.columns.tolist(),
        y_train=y_train.to_numpy(),
        y_train_pred=y_pred_train,
        y_test=y_test.to_numpy(),
        y_test_pred=y_pred_test,
    )


# -----------------------------
# Prediction for "next season"
# -----------------------------

# Prepares next-season data in same format as training set, then predicts scores
def align_and_predict_next(model, train_X_cols, next_df, numeric_feats, categorical_feats):
    """Apply training preprocessing to next season's data and predict scores."""
    available_numeric = [c for c in numeric_feats if c in next_df.columns]
    available_categorical = [c for c in categorical_feats if c in next_df.columns]

    X_next = pd.get_dummies(
        next_df[available_numeric + available_categorical],
        columns=available_categorical, drop_first=True, dtype=int
    )

    # Align columns to match training matrix; fill unseen dummies with zeros
    X_next_aligned = X_next.reindex(columns=train_X_cols, fill_value=0)
    preds = model.predict(X_next_aligned)

    out = next_df.copy()
    out["predicted_survivalscore"] = preds
    return out