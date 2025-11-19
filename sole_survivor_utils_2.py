"""
sole_survivor_utils.py

Utilities for analyzing Sole Survivor data and training a regression model
to evaluate expert initial ratings and predict overall survival scores.

- Standardizes column names
- Auto-detects numeric vs categorical features
- Correlation diagnostics (linearity to target; independence among predictors)
- Linear Regression training & evaluation
- Optional Feature Selection (K-Best, Mutual Info, RFE, or Lasso-based)
- Plotting helpers (Pred vs Actual, Residuals, Corr Heatmap)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    RFE,
    SelectFromModel,
)

# -----------------------------
# Small helpers
# -----------------------------

# Standardizes DataFrame column names (lowercase, no spaces/dashes)
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase headers; remove spaces and dashes."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    return df


# Calculates Root Mean Squared Error between true and predicted values
def rmse(y_true, y_pred) -> float:
    """Return the root mean squared error between actual and predicted values."""
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


# -----------------------------
# Feature diagnostics
# -----------------------------

# Identifies which columns are numeric vs categorical (ignores the target column)
def detect_feature_types(df: pd.DataFrame, target: str):
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
def correlations_with_target(df: pd.DataFrame, target: str, features: list[str]) -> pd.Series:
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
def predictor_independence_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Predictor-predictor correlation matrix for numeric features."""
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return pd.DataFrame()
    return df[numeric].corr()


# Finds pairs of numeric predictors that are highly correlated (above threshold)
def find_high_corr_pairs(corr_mat: pd.DataFrame, threshold: float = 0.8):
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
def plot_corr_heatmap(corr_mat: pd.DataFrame, title: str):
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
def plot_pred_vs_actual(y_true, y_pred, title: str, target_name: str = "Survival Score"):
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
def plot_residuals(y_true, y_pred, title: str, target_name: str = "Survival Score"):
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

# Simple container for model, metrics, and predicted values - simpler in the long run...
@dataclass
class ModelArtifacts:
    model: LinearRegression
    train_metrics: dict
    test_metrics: dict
    feature_columns: list            # pre-selection columns (after one-hot)
    y_train: np.ndarray
    y_train_pred: np.ndarray
    y_test: np.ndarray
    y_test_pred: np.ndarray
    selector: object | None = None   # fitted selector (if any)
    selected_features: list | None = None  # names of columns kept (when available)


# Builds X (features) and y (target) with categorical encoding if necessary
def build_design_matrices(df: pd.DataFrame, target: str, numeric_feats: list[str] | None = None, categorical_feats: list[str] | None = None):
    """Build X (with one-hot encoded categoricals) and y. Auto-detect features when None."""
    if numeric_feats is None or categorical_feats is None:
        auto_num, auto_cat = detect_feature_types(df, target)
        if numeric_feats is None:
            numeric_feats = auto_num
        if categorical_feats is None:
            categorical_feats = auto_cat

    # Drop identifier-like categorical features before encoding (prevents leakage)
    safe_cat_feats = [
        c for c in categorical_feats
        if c.lower() not in ("name", "id", "player", "contestant")
    ]

    # Optional: notify developer when identifiers are dropped
    dropped = set(categorical_feats) - set(safe_cat_feats)
    if dropped:
        print(f" Dropped identifier-like categorical columns: {list(dropped)}")

    # Build data subset for modeling
    cols = [c for c in (numeric_feats + safe_cat_feats + [target]) if c in df.columns]
    data = df[cols].copy()

    X = pd.get_dummies(
        data[numeric_feats + safe_cat_feats],
        columns=safe_cat_feats, drop_first=True, dtype=int
    )
    y = pd.to_numeric(data[target], errors="coerce")
    return X, y, numeric_feats, safe_cat_feats


# -----------------------------
# Feature selection
# -----------------------------

def make_feature_selector(method: str = "kbest", k: int = 10):
    """
    Return an unfitted sklearn-style selector.
      method: "kbest" (f_regression), "kbest_mi" (mutual_info), "rfe", "lasso_sfmodel"
      k: number of features to keep for k-best / RFE
    """
    m = method.lower()
    if m == "kbest":
        return SelectKBest(score_func=f_regression, k=k)
    if m == "kbest_mi":
        return SelectKBest(score_func=mutual_info_regression, k=k)
    if m == "rfe":
        return RFE(estimator=LinearRegression(), n_features_to_select=k, step=1)
    if m == "lasso_sfmodel":
        # Embedded selection using LassoCV; threshold='median' keeps ~50% by magnitude
        return SelectFromModel(estimator=LassoCV(cv=5, random_state=99), threshold="median")
    raise ValueError(f"Unknown feature selection method: {method}")


# Trains a linear regression model, evaluates it, and returns metrics + predictions
def fit_evaluate_linear_regression(
    X, y, test_size=0.25, random_state=99, selector=None
):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np

    def _rmse(y_true, y_pred):
        return float(mean_squared_error(y_true, y_pred) ** 0.5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    selected_columns = None
    if selector is not None:
        selector.fit(X_train, y_train)          # fit on TRAIN ONLY
        X_train = selector.transform(X_train)
        X_test  = selector.transform(X_test)
        try:
            support = selector.get_support()
            selected_columns = X.columns[support].tolist()
        except Exception:
            selected_columns = None

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    from dataclasses import dataclass
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
        selector: object | None = None
        selected_features: list | None = None

    return ModelArtifacts(
        model=model,
        train_metrics={"r2": float(r2_score(y_train, y_pred_train)), "rmse": _rmse(y_train, y_pred_train)},
        test_metrics={"r2": float(r2_score(y_test,  y_pred_test )), "rmse": _rmse(y_test,  y_pred_test )},
        feature_columns=X.columns.tolist() if hasattr(X, "columns") else [],
        y_train=getattr(y_train, "to_numpy", lambda: y_train)(),
        y_train_pred=y_pred_train,
        y_test=getattr(y_test, "to_numpy", lambda: y_test)(),
        y_test_pred=y_pred_test,
        selector=selector,
        selected_features=selected_columns,
    )


# -----------------------------
# Prediction for "next season"
# -----------------------------

def align_and_predict_next(
    model: LinearRegression,
    train_X_cols: list[str],
    next_df: pd.DataFrame,
    numeric_feats: list[str],
    categorical_feats: list[str],
    selector=None,
) -> pd.DataFrame:
    """Apply training preprocessing to next season's data and predict scores."""
    # Defensive copies
    next_df = next_df.copy()
    available_numeric = [c for c in numeric_feats if c in next_df.columns]
    available_categorical = [c for c in categorical_feats if c in next_df.columns]

    # Build design; handle case with zero available predictors
    if not available_numeric and not available_categorical:
        # Create an all-zeros frame with the training columns
        X_next_aligned = pd.DataFrame(0, index=next_df.index, columns=train_X_cols)
    else:
        X_next = pd.get_dummies(
            next_df[available_numeric + available_categorical],
            columns=available_categorical, drop_first=True, dtype=int
        )
        # Align to training columns (pre-selection)
        X_next_aligned = X_next.reindex(columns=train_X_cols, fill_value=0)

    # Apply selector if present (and fitted)
    if selector is not None:
        try:
            X_next_aligned = selector.transform(X_next_aligned)
        except Exception as e:
            # Fall back to no selection if something odd happens, but don't return None
            print(f"⚠️ Feature selector transform failed on next data: {e}. Skipping selector.")
            # keep X_next_aligned as-is

    # Predict
    try:
        preds = model.predict(X_next_aligned)
    except Exception as e:
        raise RuntimeError(f"Prediction failed. X_next_aligned shape={getattr(X_next_aligned, 'shape', None)}") from e

    out = next_df.copy()
    out["predicted_survivalscore"] = preds
    return out