import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

# Keep your existing pretty-print helper (useful & generic)
def print_1d_data_summary(data_1d, name=None):
    arr = np.array(data_1d)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.flatten()
    first = ", ".join(f"{x:7.3f}" for x in arr[:5])
    last  = ", ".join(f"{x:7.3f}" for x in arr[-5:])
    label = f"{name}: " if name else ""
    print(f"{label}[{first}, ..., {last}]")


def simple_linear_regression(cherry_tree_df, create_testing_set):
    """
    One predictor (Diam) -> one response (Height).
    Helpers for splitting, summarizing, and plotting are INLINED here.
    """
    # --- inline split ---
    X = cherry_tree_df[["Diam"]].values
    y = cherry_tree_df["Height"].values
    if create_testing_set:
        X_tr, X_te, y_tr, y_te = ms.train_test_split(X, y, test_size=0.25, random_state=42)
    else:
        X_tr, y_tr = X, y
        X_te = y_te = None

    # --- fit & predict (no top-level helpers) ---
    model = lm.LinearRegression().fit(X_tr, y_tr)
    yhat_tr = model.predict(X_tr)

    # --- inline summary helper ---
    def summarize_set(name, X=None, y_pred=None, y_true=None):
        print(name + ":")
        if X is not None:      print_1d_data_summary(X, "predictors")
        if y_pred is not None: print_1d_data_summary(y_pred, "prediction")
        if y_true is not None: print_1d_data_summary(y_true, "response")

    summarize_set("Training", X=X_tr, y_pred=yhat_tr, y_true=y_tr)

    # --- inline plotting for 1D regression (scatter + best-fit line) ---
    def plot_simple(X_in, y_true_in, y_pred_in, title):
        Xf = np.array(X_in).reshape(-1)
        order = np.argsort(Xf)
        plt.figure()
        plt.scatter(Xf, y_true_in, label="Data", color="blue", alpha=0.85)
        plt.plot(Xf[order], np.array(y_pred_in)[order], label="Best Fit Line", color="red")
        plt.xlabel("Diam"); plt.ylabel("Height"); plt.title(title); plt.legend()
        plt.tight_layout(); plt.show()

    plot_simple(X_tr, y_tr, yhat_tr, "Linear Regression: Diam vs Height (Training Data)")

    if X_te is not None:
        yhat_te = model.predict(X_te)
        summarize_set("Testing", X=X_te, y_pred=yhat_te, y_true=y_te)
        plot_simple(X_te, y_te, yhat_te, "Linear Regression: Diam vs Height (Testing Data)")


def multiple_linear_regression(cherry_tree_df, create_testing_set, one_hot_encode):
    """
    Multiple predictors -> one response (Volume).
    For plotting, show True vs Predicted with y=x (since X is multi-D).
    All helpers are INLINED here.
    """
    # --- build predictors (with optional one-hot) ---
    if one_hot_encode:
        season_dummies = pd.get_dummies(cherry_tree_df["Season"], prefix="Season")
        X = pd.concat([cherry_tree_df[["Diam", "Height"]], season_dummies], axis=1).values
    else:
        X = cherry_tree_df[["Diam", "Height"]].values
    y = cherry_tree_df["Volume"].values

    # --- inline split ---
    if create_testing_set:
        X_tr, X_te, y_tr, y_te = ms.train_test_split(X, y, test_size=0.25, random_state=42)
    else:
        X_tr, y_tr = X, y
        X_te = y_te = None

    # --- fit & predict ---
    model = lm.LinearRegression().fit(X_tr, y_tr)
    yhat_tr = model.predict(X_tr)

    # --- inline summary helper ---
    def summarize_set(name, y_pred=None, y_true=None):
        print(name + ":")
        if y_pred is not None: print_1d_data_summary(y_pred, "prediction")
        if y_true is not None: print_1d_data_summary(y_true, "response")

    summarize_set("Training", y_pred=yhat_tr, y_true=y_tr)

    # --- inline plotting: True vs Predicted with y=x line ---
    def plot_pred_vs_true(y_true_in, y_pred_in, title):
        plt.figure()
        plt.scatter(y_true_in, y_pred_in, alpha=0.8, color="blue", label="Predictions")
        lo = float(min(np.min(y_true_in), np.min(y_pred_in)))
        hi = float(max(np.max(y_true_in), np.max(y_pred_in)))
        plt.plot([lo, hi], [lo, hi], "--", color="red", label="Ideal (y=x)")
        plt.xlabel("True"); plt.ylabel("Predicted"); plt.title(title); plt.legend()
        plt.tight_layout(); plt.show()

    plot_pred_vs_true(y_tr, yhat_tr, "Multiple LR: Volume — True vs Predicted (Training)")

    if X_te is not None:
        yhat_te = model.predict(X_te)
        summarize_set("Testing", y_pred=yhat_te, y_true=y_te)
        plot_pred_vs_true(y_te, yhat_te, "Multiple LR: Volume — True vs Predicted (Testing)")


def main():
    cherry_tree_df = pd.read_csv("1_linear_regression/CherryTree.csv")

    simple_linear_regression(cherry_tree_df, False)
    # simple_linear_regression(cherry_tree_df, True)
    # multiple_linear_regression(cherry_tree_df, False, False)
    # multiple_linear_regression(cherry_tree_df, False, True)
    # multiple_linear_regression(cherry_tree_df, True, False)

if __name__ == "__main__":
    main()
