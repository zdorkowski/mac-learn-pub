# Titanic Classification Analysis - Zdorkowski BIG DATA Edition

# Step 1 — Import libraries and load dataset

import pandas as pd
import numpy as np
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


# Load the dataset
FILE_PATH = "Titanic-Dataset.csv"
titanic = pd.read_csv(FILE_PATH)
# Preview structure
print("Shape:", titanic.shape)
titanic.head()

# Step 2 — Clean missing data

# Replace Cabin with 0 so it doesn't delete 77% of the dataset
titanic["Cabin"] = 0

# Count missing values before removal
print("Missing values per column before cleanup:")
print(titanic.isna().sum())

# Drop any remaining rows containing NA values
titanic_clean = titanic.dropna()

# Confirm removal
print("\nShape before:", titanic.shape)
print("Shape after:", titanic_clean.shape)

# Step 3 — Logistic Regression Classification

# Select predictors and response (initial candidates)
predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
response = "Survived"

# Keep only predictors with |correlation to Survived| >= 0.1
corr_with_y = titanic_clean[predictors + [response]].corr()[response].drop(response)
print("Correlation of predictors with Survived:")
print(corr_with_y, "\n")

# Apply the 0.1 threshold (absolute value)
selected_predictors = [col for col, val in corr_with_y.items() if abs(val) >= 0.1]
print("Keeping predictors with |corr| >= 0.1:")
print(selected_predictors, "\n")

# Update the predictors list to only include those
predictors = selected_predictors

# Now build X and y using the filtered predictors
X = titanic_clean[predictors]
y = titanic_clean[response]

# # Split into train/test sets (use random_state=1)
# X_train, X_test, y_train, y_test = ms.train_test_split(
#     X, y, test_size=0.2, random_state=1
# )

# Use stratify=y so the train/test sets keep the same Survived proportionally

X_train, X_test, y_train, y_test = ms.train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Build and fit logistic regression model
log_reg = lm.LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict
y_pred = log_reg.predict(X_test)

# Evaluate performance
print("=== Classification Report ===")
print(metrics.classification_report(y_test, y_pred, zero_division=0))

print("=== Confusion Matrix ===")
print(metrics.confusion_matrix(y_test, y_pred))

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# BONUS — Exploratory plots

# Age vs. Survival
plt.figure(figsize=(6,4))
plt.scatter(titanic_clean["Age"], titanic_clean["Survived"], alpha=0.5)
plt.title("Age vs Survival")
plt.xlabel("Age")
plt.ylabel("Survived (1 = Yes, 0 = No)")
plt.grid(True)
plt.show()

# Fare vs. Survival
plt.figure(figsize=(6,4))
plt.scatter(titanic_clean["Fare"], titanic_clean["Survived"], alpha=0.5, color='orange')
plt.title("Fare vs Survival")
plt.xlabel("Fare")
plt.ylabel("Survived (1 = Yes, 0 = No)")
plt.grid(True)
plt.show()

# Passenger Class distribution
plt.figure(figsize=(5,4))
titanic_clean["Pclass"].value_counts().sort_index().plot(
    kind="bar", color=["#3498db", "#2ecc71", "#e74c3c"], edgecolor="black"
)
plt.title("Passenger Class Distribution")
plt.xlabel("Pclass (1 = 1st, 2 = 2nd, 3 = 3rd)")
plt.ylabel("Count")
plt.show()

# BONUS BONUS — Visualize Logistic Regression Coefficients

coef_table = pd.DataFrame({
    "Feature": predictors,
    "Coefficient": log_reg.coef_[0]
}).sort_values(by="Coefficient")

plt.figure(figsize=(6,4))
plt.barh(coef_table["Feature"], coef_table["Coefficient"], color="teal", edgecolor="black")
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.grid(True)
plt.show()

coef_table

# Step 4 — Check linearity and independence among predictors

# --- 4a: Correlation matrix (numeric features only)
corr = titanic_clean[predictors].corr(numeric_only=True)
print("Correlation Matrix:\n", corr, "\n")

plt.figure(figsize=(6,5))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.title("Correlation Matrix (Predictors)")
plt.colorbar()
plt.xticks(range(len(predictors)), predictors, rotation=45)
plt.yticks(range(len(predictors)), predictors)
plt.tight_layout()
plt.show()

# --- 4b: Check for strong correlation (multi-collinearity)
threshold = 0.8
high_corr = [(a,b,corr.loc[a,b]) for a in predictors for b in predictors if a!=b and abs(corr.loc[a,b])>threshold]
if high_corr:
    print("Highly correlated features (possible multicollinearity):")
    for a,b,val in high_corr:
        print(f"{a} ↔ {b}: {val:.3f}")
else:
    print("No predictors exceed ±0.8 correlation — independence assumption reasonable.")

# --- 4c: Quick Variance Inflation Factor (VIF) check
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

X_vif = titanic_clean[predictors].assign(Intercept=1)
vif_data = pd.DataFrame({
    "Feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print("\nVariance Inflation Factors (VIF):\n", vif_data)

# Step 5 — Inspect predicted values

print("Unique prediction values:", np.unique(y_pred))
print("Prediction sample (first 20):", y_pred[:20])

# # Confirm predictions are strictly binary
# if set(np.unique(y_pred)) == {0,1}:
#     print("All predictions are 0s and 1s — consistent with binary classification.")
# else:
#     print("Unexpected non-binary predictions detected.")

# Step 6 — Compare to an all-ones predictor

# Create a all-ones prediction array: all passengers predicted to survive
all_ones = np.ones_like(y_test)

# Evaluate this all-ones model
print("=== all-ones All-Ones Classifier Performance ===")
print(metrics.classification_report(y_test, all_ones, zero_division=0))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, all_ones))

# Compare to logistic regression accuracy
acc_lr = metrics.accuracy_score(y_test, y_pred)
acc_ones = metrics.accuracy_score(y_test, all_ones)
print(f"\nLogistic Regression Accuracy: {acc_lr:.4f}")
print(f"All-Ones Accuracy: {acc_ones:.4f}")

if acc_ones >= acc_lr:
    print("The all-ones model performs as well or better — suggests imbalance or weak separation.")
else:
    print("Logistic Regression outperforms the all-ones model.")

# Task 7 — Encode Sex (binary) and Embarked (3 categories)

# Inspect raw categories
print("Unique Sex values:", titanic_clean["Sex"].dropna().unique())
print("Unique Embarked values:", titanic_clean["Embarked"].dropna().unique())

# Binary encode Sex: male=1, female=0
titanic_clean = titanic_clean.copy()  # make an explicit copy once
titanic_clean.loc[:, "Sex_bin"] = titanic_clean["Sex"].map({"male": 1, "female": 0}).astype("int8")

# One-hot encode Embarked (3 categories -> 2 dummy columns with drop_first=True)
emb_dum = pd.get_dummies(titanic_clean["Embarked"], prefix="Embarked", drop_first=True)

# Join back to the working frame
titanic_model = titanic_clean.join(emb_dum)

# Verify new columns
print("New columns added:", [c for c in titanic_model.columns if c.startswith("Sex_") or c.startswith("Embarked_")] + ["Sex_bin"])
titanic_model.head()

# Task 8 — Re-run with new features (Sex_bin and Embarked dummies)

# Base numeric predictors from earlier
base_predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

# Grab the Embarked dummy column names (2 columns because of drop_first=True)
emb_cols = [c for c in titanic_model.columns if c.startswith("Embarked_")]

predictors_v2 = base_predictors + ["Sex_bin"] + emb_cols
response = "Survived"

X2 = titanic_model[predictors_v2]
y2 = titanic_model[response]

# Use the fair split (stratify=y2) and the same random_state=1 per earlier instruction
X2_train, X2_test, y2_train, y2_test = ms.train_test_split(
    X2, y2, test_size=0.2, random_state=1, stratify=y2
)

# First run with default max_iter to observe potential convergence warning
log_reg_v2 = lm.LogisticRegression(max_iter=1000)  # may warn about convergence
log_reg_v2.fit(X2_train, y2_train)

y2_pred = log_reg_v2.predict(X2_test)

print("=== V2 Classification Report (default max_iter) ===")
print(metrics.classification_report(y2_test, y2_pred, zero_division=0))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y2_test, y2_pred))

print("Accuracy:", metrics.accuracy_score(y2_test, y2_pred))

# Task 9 — Raise iteration cap to ensure convergence

log_reg_v2_big = lm.LogisticRegression(max_iter=100_000)
log_reg_v2_big.fit(X2_train, y2_train)

y2_pred_big = log_reg_v2_big.predict(X2_test)
y2_prob_big = log_reg_v2_big.predict_proba(X2_test)[:, 1]

print("=== V2 (max_iter=100000) Classification Report ===")
print(metrics.classification_report(y2_test, y2_pred_big, zero_division=0))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y2_test, y2_pred_big))
print("Accuracy:", metrics.accuracy_score(y2_test, y2_pred_big))

# Task 10 — Oversample with RandomOverSampler + Cross-Validation

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

# 10a. Define a stratified 5-fold CV splitter (keeps class balance in each fold)
cv = ms.StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# 10b. Baseline CV (no oversampling)
baseline_clf = lm.LogisticRegression(max_iter=100_000)
baseline_scores = ms.cross_val_score(
    baseline_clf,
    X2,
    y2,
    cv=cv,
    scoring="accuracy"
    # scoring="f1"  # alternative metric if desired
)

print("\n=== Baseline Cross-Validation (no oversampling) ===")
print("Fold accuracies:", np.round(baseline_scores, 3))
print("Mean accuracy:   {:.3f}".format(baseline_scores.mean()))
print("Std dev:         {:.3f}".format(baseline_scores.std()))

# 10c. CV with RandomOverSampler inside a Pipeline
oversampled_pipeline = Pipeline(steps=[
    ("oversample", RandomOverSampler(random_state=1)),
    ("log_reg", lm.LogisticRegression(max_iter=100_000))
])

oversampled_scores = ms.cross_val_score(
    oversampled_pipeline,
    X2,
    y2,
    cv=cv,
    scoring="accuracy"
    # scoring="f1"  # alternative metric if desired
)

print("\n=== Cross-Validation with RandomOverSampler ===")
print("Fold accuracies:", np.round(oversampled_scores, 3))
print("Mean accuracy:   {:.3f}".format(oversampled_scores.mean()))
print("Std dev:         {:.3f}".format(oversampled_scores.std()))

# Quick comparison
print("\n=== Comparison ===")
print("Baseline mean CV accuracy:   {:.3f}".format(baseline_scores.mean()))
print("Oversampled mean CV accuracy:{:.3f}".format(oversampled_scores.mean()))

# Visual comparison via boxplot
plt.figure(figsize=(6,4))
plt.boxplot([baseline_scores, oversampled_scores], tick_labels=["Baseline", "Oversampled"])
plt.title("Cross-Validation Accuracy Comparison")
plt.ylabel("Score")
plt.grid(True)
plt.show()

# Task 11 — Sensitivity and Specificity Comparison Before/After Oversampling

# Helper function to calculate sensitivity and specificity
def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)   # Recall for class 1
    specificity = tn / (tn + fp)   # True negative rate

    return sensitivity, specificity


# -------------------------
# BEFORE balancing
# -------------------------
baseline_model = lm.LogisticRegression(max_iter=100000)
baseline_model.fit(X2_train, y2_train)
y_base_pred = baseline_model.predict(X2_test)

sens_before, spec_before = sensitivity_specificity(y2_test, y_base_pred)

print("\n=== BEFORE Oversampling ===")
print(f"Sensitivity (Recall for 1): {sens_before:.3f}")
print(f"Specificity (True Negative Rate): {spec_before:.3f}")


# -------------------------
# AFTER balancing
# -------------------------
oversampled_pipeline.fit(X2_train, y2_train)
y_over_pred = oversampled_pipeline.predict(X2_test)

sens_after, spec_after = sensitivity_specificity(y2_test, y_over_pred)

print("\n=== AFTER Oversampling ===")
print(f"Sensitivity (Recall for 1): {sens_after:.3f}")
print(f"Specificity (True Negative Rate): {spec_after:.3f}")


# -------------------------
# Comparison summary
# -------------------------
print("\n=== CHANGE DUE TO BALANCING ===")
print(f"Sensitivity change: {sens_before:.3f} → {sens_after:.3f}")
print(f"Specificity change: {spec_before:.3f} → {spec_after:.3f}")

# Task 12 — Write all relevant outputs to a .txt file

output_file = "Titanic_Classification_Results_BIGDATA.txt"

with open(output_file, "w", encoding="utf-8") as f:

    f.write("TITANIC CLASSIFICATION ANALYSIS — RESULTS\n")
    f.write("=" * 50 + "\n\n")

    # --- Dataset information ---
    f.write("DATASET INFO\n")
    f.write(f"Original rows: {len(titanic)}\n")
    f.write(f"Rows after dropna(): {len(titanic_clean)}\n\n")

    # --- Task 3: Baseline Logistic Regression ---
    f.write("TASK 3 — BASELINE LOGISTIC REGRESSION (numeric features)\n")
    f.write(f"Accuracy: {acc_lr:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(metrics.confusion_matrix(y_test, y_pred)) + "\n\n")

    # --- Task 6: all-ones All-Ones ---
    f.write("TASK 6 — all-ones ALL-ONES CLASSIFIER\n")
    f.write(f"Accuracy: {acc_ones:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(metrics.confusion_matrix(y_test, all_ones)) + "\n\n")

    # --- Task 9: Full model (Sex + Embarked)
    f.write("TASK 9 — LOGISTIC REGRESSION WITH Sex + Embarked\n")
    f.write(f"Accuracy: {metrics.accuracy_score(y2_test, y2_pred_big):.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(metrics.confusion_matrix(y2_test, y2_pred_big)) + "\n\n")

    # --- Task 10: Cross-validation
    f.write("TASK 10 — CROSS-VALIDATION RESULTS (5-fold stratified)\n")
    f.write("Baseline fold accuracies:\n")
    f.write(str(np.round(baseline_scores, 4)) + "\n")
    f.write(f"Baseline mean accuracy: {baseline_scores.mean():.4f}\n")
    f.write(f"Baseline std dev: {baseline_scores.std():.4f}\n\n")

    f.write("Oversampled fold accuracies:\n")
    f.write(str(np.round(oversampled_scores, 4)) + "\n")
    f.write(f"Oversampled mean accuracy: {oversampled_scores.mean():.4f}\n")
    f.write(f"Oversampled std dev: {oversampled_scores.std():.4f}\n\n")

    f.write(f"Change in mean accuracy (oversampled - baseline): "
            f"{(oversampled_scores.mean() - baseline_scores.mean()):.4f}\n\n")

    # --- Task 11: Sensitivity / Specificity
    f.write("TASK 11 — SENSITIVITY & SPECIFICITY\n")
    f.write("Before oversampling:\n")
    f.write(f"  Sensitivity (Recall for class 1): {sens_before:.4f}\n")
    f.write(f"  Specificity (True Negative Rate): {spec_before:.4f}\n\n")

    f.write("After oversampling:\n")
    f.write(f"  Sensitivity (Recall for class 1): {sens_after:.4f}\n")
    f.write(f"  Specificity (True Negative Rate): {spec_after:.4f}\n\n")

    # --- Final note
    f.write("END OF REPORT\n")

print(f"Task 12 complete! Results saved to: {output_file}")