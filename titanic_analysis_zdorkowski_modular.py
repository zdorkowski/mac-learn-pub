# Titanic Classification Analysis - Zdorkowski - MODULAR Edition

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

FILE_PATH = "Titanic-Dataset.csv"
OUTPUT_FILE = "Titanic_Classification_Results.txt"


def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn), tn / (tn + fp)


def main():
    # Load and clean
    titanic = pd.read_csv(FILE_PATH)
    print("Shape:", titanic.shape)
    print("Missing values per column before cleanup:")
    print(titanic.isna().sum())
    titanic_clean = titanic.dropna()
    print("\nShape before:", titanic.shape)
    print("Shape after:", titanic_clean.shape)

    # Task 3 — numeric baseline with corr filter
    base_predictors = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    target = "Survived"
    corr_with_y = titanic_clean[base_predictors + [target]].corr()[target].drop(target)
    print("Correlation of predictors with Survived:")
    print(corr_with_y, "\n")
    predictors = [c for c, v in corr_with_y.items() if abs(v) >= 0.1]
    print("Keeping predictors with |corr| >= 0.1:")
    print(predictors, "\n")

    X_num = titanic_clean[predictors]
    y = titanic_clean[target]
    X_train, X_test, y_train, y_test = ms.train_test_split(
        X_num, y, test_size=0.2, random_state=1, stratify=y
    )
    log_reg = lm.LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print("=== Baseline Classification Report (numeric only) ===")
    print(metrics.classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    acc_lr = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", acc_lr)

    # Bonus plots
    plt.figure(figsize=(6, 4))
    plt.scatter(titanic_clean["Age"], titanic_clean["Survived"], alpha=0.5)
    plt.title("Age vs Survival")
    plt.xlabel("Age")
    plt.ylabel("Survived")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.scatter(titanic_clean["Fare"], titanic_clean["Survived"], alpha=0.5)
    plt.title("Fare vs Survival")
    plt.xlabel("Fare")
    plt.ylabel("Survived")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 4))
    titanic_clean["Pclass"].value_counts().sort_index().plot(kind="bar", edgecolor="black")
    plt.title("Passenger Class Distribution")
    plt.xlabel("Pclass")
    plt.ylabel("Count")
    plt.show()

    coef_table = pd.DataFrame(
        {"Feature": predictors, "Coefficient": log_reg.coef_[0]}
    ).sort_values("Coefficient")
    print("\nCoefficient table:")
    print(coef_table, "\n")
    plt.figure(figsize=(6, 4))
    plt.barh(coef_table["Feature"], coef_table["Coefficient"], edgecolor="black")
    plt.title("Logistic Regression Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.grid(True)
    plt.show()

    # Step 4 — Correlation matrix + VIF
    corr = titanic_clean[predictors].corr(numeric_only=True)
    print("Correlation Matrix:\n", corr, "\n")
    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.title("Correlation Matrix (Predictors)")
    plt.colorbar()
    plt.xticks(range(len(predictors)), predictors, rotation=45)
    plt.yticks(range(len(predictors)), predictors)
    plt.tight_layout()
    plt.show()

    threshold = 0.8
    high_corr = [
        (a, b, corr.loc[a, b])
        for a in predictors for b in predictors
        if a != b and abs(corr.loc[a, b]) > threshold
    ]
    if high_corr:
        print("Highly correlated features (possible multicollinearity):")
        for a, b, v in high_corr:
            print(f"{a} ↔ {b}: {v:.3f}")
    else:
        print("No predictors exceed ±0.8 correlation — independence reasonable.")

    X_vif = titanic_clean[predictors].assign(Intercept=1)
    vif = pd.DataFrame(
        {
            "Feature": X_vif.columns,
            "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
        }
    )
    print("\nVariance Inflation Factors (VIF):\n", vif, "\n")

    # Step 5 — Inspect predictions
    print("Unique prediction values:", np.unique(y_pred))
    print("Prediction sample (first 20):", y_pred[:20])

    # Step 6 — all-ones comparison
    all_ones = np.ones_like(y_test)
    print("\n=== All-Ones Classifier ===")
    print(metrics.classification_report(y_test, all_ones, zero_division=0))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, all_ones))
    acc_ones = metrics.accuracy_score(y_test, all_ones)
    print(f"\nLogistic Regression Accuracy: {acc_lr:.4f}")
    print(f"All-Ones Accuracy: {acc_ones:.4f}")
    if acc_ones >= acc_lr:
        print("All-ones matches or beats logistic → imbalance / weak separation.")
    else:
        print("Logistic regression outperforms all-ones.")

    # Task 7 — encode Sex + Embarked
    print("Unique Sex values:", titanic_clean["Sex"].dropna().unique())
    print("Unique Embarked values:", titanic_clean["Embarked"].dropna().unique())
    model_df = titanic_clean.copy()
    model_df["Sex_bin"] = model_df["Sex"].map({"male": 1, "female": 0}).astype("int8")
    emb_dum = pd.get_dummies(model_df["Embarked"], prefix="Embarked", drop_first=True)
    model_df = model_df.join(emb_dum)
    emb_cols = [c for c in model_df.columns if c.startswith("Embarked_")]
    print("New columns added:", emb_cols + ["Sex_bin"])
    print(model_df.head())

    # Task 8/9 — full model with Sex + Embarked
    predictors_v2 = base_predictors + ["Sex_bin"] + emb_cols
    X2 = model_df[predictors_v2]
    y2 = model_df[target]
    X2_train, X2_test, y2_train, y2_test = ms.train_test_split(
        X2, y2, test_size=0.2, random_state=1, stratify=y2
    )
    log_reg_v2 = lm.LogisticRegression(max_iter=100_000)
    log_reg_v2.fit(X2_train, y2_train)
    y2_pred = log_reg_v2.predict(X2_test)
    print("=== V2 (Sex + Embarked) ===")
    print(metrics.classification_report(y2_test, y2_pred, zero_division=0))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y2_test, y2_pred))
    acc_v2 = metrics.accuracy_score(y2_test, y2_pred)
    print("Accuracy:", acc_v2)

    # Task 10 — CV baseline vs oversampled
    cv = ms.StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    baseline_scores = ms.cross_val_score(
        lm.LogisticRegression(max_iter=100_000), X2, y2, cv=cv, scoring="accuracy"
    )
    print("\n=== Baseline CV (no oversampling) ===")
    print("Fold accuracies:", np.round(baseline_scores, 3))
    print("Mean accuracy:   {:.3f}".format(baseline_scores.mean()))
    print("Std dev:         {:.3f}".format(baseline_scores.std()))

    oversampled_pipeline = Pipeline(
        steps=[
            ("oversample", RandomOverSampler(random_state=1)),
            ("log_reg", lm.LogisticRegression(max_iter=100_000)),
        ]
    )
    oversampled_scores = ms.cross_val_score(
        oversampled_pipeline, X2, y2, cv=cv, scoring="accuracy"
    )
    print("\n=== CV with RandomOverSampler ===")
    print("Fold accuracies:", np.round(oversampled_scores, 3))
    print("Mean accuracy:   {:.3f}".format(oversampled_scores.mean()))
    print("Std dev:         {:.3f}".format(oversampled_scores.std()))

    print("\n=== Comparison ===")
    print("Baseline mean CV accuracy:   {:.3f}".format(baseline_scores.mean()))
    print("Oversampled mean CV accuracy:{:.3f}".format(oversampled_scores.mean()))

    plt.figure(figsize=(6, 4))
    plt.boxplot([baseline_scores, oversampled_scores], tick_labels=["Baseline", "Oversampled"])
    plt.title("Cross-Validation Accuracy Comparison")
    plt.ylabel("Score")
    plt.grid(True)
    plt.show()

    # Task 11 — Sensitivity / Specificity before/after oversampling
    baseline_model = lm.LogisticRegression(max_iter=100_000)
    baseline_model.fit(X2_train, y2_train)
    y_base_pred = baseline_model.predict(X2_test)
    sens_before, spec_before = sensitivity_specificity(y2_test, y_base_pred)

    oversampled_pipeline.fit(X2_train, y2_train)
    y_over_pred = oversampled_pipeline.predict(X2_test)
    sens_after, spec_after = sensitivity_specificity(y2_test, y_over_pred)

    print("\n=== BEFORE Oversampling ===")
    print(f"Sensitivity (Recall for 1): {sens_before:.3f}")
    print(f"Specificity (True Negative Rate): {spec_before:.3f}")
    print("\n=== AFTER Oversampling ===")
    print(f"Sensitivity (Recall for 1): {sens_after:.3f}")
    print(f"Specificity (True Negative Rate): {spec_after:.3f}")
    print("\n=== CHANGE DUE TO BALANCING ===")
    print(f"Sensitivity change: {sens_before:.3f} → {sens_after:.3f}")
    print(f"Specificity change: {spec_before:.3f} → {spec_after:.3f}")

    # Task 12 — write report
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("TITANIC CLASSIFICATION ANALYSIS — RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write("DATASET INFO\n")
        f.write(f"Original rows: {len(titanic)}\n")
        f.write(f"Rows after dropna(): {len(titanic_clean)}\n\n")

        f.write("TASK 3 — BASELINE LOGISTIC REGRESSION (numeric features)\n")
        f.write(f"Accuracy: {acc_lr:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics.confusion_matrix(y_test, y_pred)) + "\n\n")

        f.write("TASK 6 — ALL-ONES CLASSIFIER\n")
        f.write(f"Accuracy: {acc_ones:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics.confusion_matrix(y_test, all_ones)) + "\n\n")

        f.write("TASK 9 — LOGISTIC REGRESSION WITH Sex + Embarked\n")
        f.write(f"Accuracy: {acc_v2:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics.confusion_matrix(y2_test, y2_pred)) + "\n\n")

        f.write("TASK 10 — CROSS-VALIDATION RESULTS (5-fold stratified)\n")
        f.write("Baseline fold accuracies:\n")
        f.write(str(np.round(baseline_scores, 4)) + "\n")
        f.write(f"Baseline mean accuracy: {baseline_scores.mean():.4f}\n")
        f.write(f"Baseline std dev: {baseline_scores.std():.4f}\n\n")
        f.write("Oversampled fold accuracies:\n")
        f.write(str(np.round(oversampled_scores, 4)) + "\n")
        f.write(f"Oversampled mean accuracy: {oversampled_scores.mean():.4f}\n")
        f.write(f"Oversampled std dev: {oversampled_scores.std():.4f}\n\n")
        f.write(
            "Change in mean accuracy (oversampled - baseline): "
            f"{(oversampled_scores.mean() - baseline_scores.mean()):.4f}\n\n"
        )

        f.write("TASK 11 — SENSITIVITY & SPECIFICITY\n")
        f.write("Before oversampling:\n")
        f.write(f"  Sensitivity (Recall for class 1): {sens_before:.4f}\n")
        f.write(f"  Specificity (True Negative Rate): {spec_before:.4f}\n\n")
        f.write("After oversampling:\n")
        f.write(f"  Sensitivity (Recall for class 1): {sens_after:.4f}\n")
        f.write(f"  Specificity (True Negative Rate): {spec_after:.4f}\n\n")
        f.write("END OF REPORT\n")

    print(f"\nTask 12 complete! Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
