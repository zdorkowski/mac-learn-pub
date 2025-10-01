"""
cars.csv — Price Prediction Analysis

This script demonstrates:
  1) Simple Linear Regression (engine size → price)
  2) Multiple Linear Regression (numeric + one-hot encoded categorical features → price)

Outputs:
  - R² (proportion of variance in the dependent variable)
  - Root Mean Squared Error metrics for both models
  - Scatter plot with regression line for the simple model
"""

# =======================
# 1. Import libraries
# =======================
# We use pandas/numpy for data handling, matplotlib/seaborn for plotting,
# and scikit-learn for modeling and evaluation.
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# (Improvement) Keep warnings quiet only for cosmetic sklearn feature-name mismatch (now avoided anyway)
warnings.filterwarnings("ignore", message="X has feature names")

# =======================
# 2. Load and clean data
# =======================
# - Load cars.csv
# - Normalize column names
# - Convert numeric columns to numeric types
# - Drop rows missing enginesize or price (needed for regression)
df = pd.read_csv("cars.csv")

# Standardize column names to lowercase and remove - and " " chars for consistency
df.columns = df.columns.str.lower().str.replace("-", "", regex=False).str.replace(" ", "", regex=False)

# Ensure core numeric columns are numeric ('coerce' bad entries to NaN)
for col in ['price', 'enginesize', 'horsepower', 'curbweight', 'highwaympg']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows missing price or enginesize
df = df.dropna(subset=['price', 'enginesize'])

# ===================================================
# 3. SIMPLE LINEAR REGRESSION (engine size → price)
# ===================================================
# - Train/test split (75/25)
# - Fit linear model (DataFrames throughout for consistency)
# - Evaluate on test set
# - Plot regression line over scatter of all data

# (Improvement) Use DataFrames for both fit & predict to avoid feature-name warning
X = df[['enginesize']]
y = df['price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=99
)

# Fit linear regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

# Calculate metrics (R2 and Root Mean Squared Error)
r2_simple = r2_score(y_test, y_pred)
rmse_simple = mean_squared_error(y_test, y_pred) ** 0.5

# Print results
print("\n=== Simple Linear Regression (Engine Size → Price) ===")
print(f"R2: {r2_simple:.3f}")
print(f"Root Mean Squared Error: ${rmse_simple:,.2f}")

# (Improvement) Sort by x before drawing regression line to ensure a clean monotonic line
_sorted = df[['enginesize', 'price']].sort_values('enginesize')

# Plot results
plt.figure(figsize=(6, 4))
sns.scatterplot(x=_sorted['enginesize'], y=_sorted['price'], alpha=0.6)
sns.lineplot(x=_sorted['enginesize'], y=linreg.predict(_sorted[['enginesize']]), color='red')
plt.title('Engine Size vs Price (Simple Linear Regression)')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.tight_layout()
plt.show()

# ============================================================
# 4. MULTIPLE LINEAR REGRESSION (numeric + categorical → price)
# ============================================================
# - Use multiple numeric predictors
# - Feature checks (linearity & independence)
# - One-hot encode categorical columns
# - Train/test split, fit, evaluate

# Define numeric and categorical features and filter to those present in df
numeric_feats = ['enginesize', 'horsepower', 'curbweight', 'highwaympg']
numeric_feats = [c for c in numeric_feats if c in df.columns]

# (Improvement) Guard against empty numeric feature list
if not numeric_feats:
    print("\nNo numeric predictors available after filtering; skipping 4A checks.")
else:
    # ============================================================
    # 4A. FEATURE SELECTION AND ASSUMPTION CHECKS
    # ============================================================
    # This section evaluates whether the numeric predictors meet key assumptions
    # of linear regression: linearity (each predictor’s relationship with the response)
    # and independence (predictors should not be excessively correlated with each other).

    # -------------------------------
    # LINEARITY CHECK (Predictor vs. Response)
    # -------------------------------
    # Pearson’s r measures the linear relationship between each predictor and 'price'.
    # As a practical guideline, flag predictors with |r| < 0.30 as weak contributors.
    linearity_threshold = 0.30
    correlations_with_price = (
        df[numeric_feats + ['price']]
        .corr()['price']
        .drop('price')
        .sort_values(ascending=False)
    )

    print("\n=== Linearity Check: Pearson Correlation with Price ===")
    print(correlations_with_price)

    strong_predictors = correlations_with_price[correlations_with_price.abs() >= linearity_threshold].index.tolist()
    weak_predictors = correlations_with_price[correlations_with_price.abs() < linearity_threshold].index.tolist()

    print(f"\nPredictors meeting linearity threshold (|r| ≥ {linearity_threshold}): {strong_predictors}")
    print(f"Predictors below threshold (|r| < {linearity_threshold}): {weak_predictors}")

    # (Improvement) Optional: enforce linearity screen automatically if it leaves at least one predictor
    if strong_predictors:
        numeric_feats = strong_predictors
    else:
        print("No predictors meet the linearity threshold; retaining original numeric features to proceed.")

    # -------------------------------
    # INDEPENDENCE CHECK (Between Predictors)
    # -------------------------------
    # Inspect the correlation matrix of predictors to identify pairs with |r| > 0.80,
    # which indicates strong interdependence that can distort regression estimates.
    corr_pred = df[numeric_feats].corr()
    print("\n=== Predictor–Predictor Correlation Matrix ===")
    print(corr_pred)

    # (Improvement) Add a quick visual heatmap (lower triangle) to assess independence at a glance
    mask = np.triu(np.ones_like(corr_pred, dtype=bool))  # show only lower triangle
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        corr_pred, mask=mask, annot=True, fmt=".2f",
        vmin=-1, vmax=1, center=0, cmap="coolwarm"
    )
    plt.title("Predictor–Predictor Correlation (Independence)")
    plt.tight_layout()
    plt.show()

    # Identify highly correlated pairs
    high_corr_pairs = [
        (c1, c2) for c1 in corr_pred.columns for c2 in corr_pred.columns
        if c1 < c2 and abs(corr_pred.loc[c1, c2]) > 0.8
    ]

    if high_corr_pairs:
        print("\nHighly correlated predictor pairs (|r| > 0.80):")
        for pair in high_corr_pairs:
            print(f"  {pair[0]} ↔ {pair[1]} (r = {corr_pred.loc[pair[0], pair[1]]:.2f})")
        print("Consider removing one predictor from each pair to improve independence.")

        # (Improvement) Optional: automatically drop the weaker-to-price feature from each high-corr pair
        r_to_price_abs = correlations_with_price.abs()  # |r(feature, price)|
        to_drop = set()
        for a, b in high_corr_pairs:
            a_r = r_to_price_abs.get(a, 0.0)
            b_r = r_to_price_abs.get(b, 0.0)
            drop = a if a_r < b_r else b
            to_drop.add(drop)

        if to_drop:
            numeric_feats = [c for c in numeric_feats if c not in to_drop]
            print(f"Dropping (independence step; kept the stronger-to-price feature): {sorted(to_drop)}")

        # (Improvement) Robust fallback: if everything got dropped, keep the single strongest predictor
        if not numeric_feats:
            best = correlations_with_price.abs().idxmax()
            numeric_feats = [best]
            print(f"No predictors left after independence drop; keeping strongest predictor: {best}")
    else:
        print("\nNo highly correlated predictor pairs found — predictors appear independent.")

# ============================================================
# 4B. MODELING (after feature checks)
# ============================================================
# Define categorical features (if present) and filter to those present in df
categorical_feats = ['fueltype', 'aspiration', 'doornumber', 'drivewheels', 'bodystyle']
categorical_feats = [c for c in categorical_feats if c in df.columns]

# One-hot encode categoricals into dummy columns
df_encoded = pd.get_dummies(
    df[numeric_feats + categorical_feats + ['price']],
    columns=categorical_feats,
    drop_first=True,
    dtype=int
)

# Prepare data for modeling
# (Improvement) Use DataFrames end-to-end for consistency/readability
X_multi = df_encoded.drop('price', axis=1)
y_multi = df_encoded['price']

# Split, fit, predict
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.25, random_state=99
)

multi_reg = LinearRegression()
multi_reg.fit(X_train_m, y_train_m)
y_pred_m = multi_reg.predict(X_test_m)

# Calculate metrics
r2_multi = r2_score(y_test_m, y_pred_m)
rmse_multi = mean_squared_error(y_test_m, y_pred_m) ** 0.5

# Print results
print("\n=== Multiple Linear Regression (Numeric + One-Hot Encoded Categorical) ===")
print(f"R2:   {r2_multi:.3f}")
print(f"Root Mean Squared Error: ${rmse_multi:,.2f}")