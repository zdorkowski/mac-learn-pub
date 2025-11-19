import random
import math
import sys
from typing import List, Tuple

Point = Tuple[float, float]

def generate_dataset() -> List[Point]:
    # Fixed 12 points in 2D space (roughly linear with small noise)
    return [
        (1, 2.1),
        (2, 3.9),
        (3, 6.2),
        (4, 8.1),
        (5, 10.2),
        (6, 12.2),
        (7, 13.9),
        (8, 16.1),
        (9, 18.0),
        (10, 20.2),
        (11, 22.1),
        (12, 24.2),
    ]


def split_dataset(data: List[Point], test_size: int = 4, seed: int = 42) -> Tuple[List[Point], List[Point]]:
    if test_size <= 0 or test_size >= len(data):
        raise ValueError("test_size must be between 1 and len(data)-1")
    rnd = random.Random(seed)
    shuffled = data[:]
    rnd.shuffle(shuffled)
    test = shuffled[:test_size]
    train = shuffled[test_size:]
    return train, test


def fit_ols_line(points: List[Point]) -> Tuple[float, float]:
    if len(points) < 2:
        raise ValueError("Need at least two points to fit a line")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    # Compute slope (beta1) and intercept (beta0)
    num = sum((x - mean_x) * (y - mean_y) for x, y in points)
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        raise ValueError("Variance of x is zero; cannot fit a unique line")
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


def predict(slope: float, intercept: float, x: float) -> float:
    return slope * x + intercept


def main():
    data = generate_dataset()
    test_size = 4
    seed = 2025
    rnd = random.Random(seed)
    shuffled = data[:]
    rnd.shuffle(shuffled)
    folds = [shuffled[i:i + test_size] for i in range(0, len(shuffled), test_size)]
    # Moved rmse_values initialization here rather than inside the loop
    rmse_values = []
    for cross_validation_counter, test in enumerate(folds, start=1):
        train = [p for p in shuffled if p not in test]
        slope, intercept = fit_ols_line(train)
        print(f"--- Cross Validation Iteration {cross_validation_counter} ---")
        print(f"Fitted line: y = {slope:.6f} * x + {intercept:.6f}\n")
        residuals = []
        for x, y in test:
            y_hat = predict(slope, intercept, x)
            resid = y - y_hat
            residuals.append(resid)
        mse = sum(r ** 2 for r in residuals) / len(residuals)
        rmse = math.sqrt(mse)
        print(f"\nRoot Mean Squared Error (RMSE) on test set: {rmse:.6f}\n")
        rmse_values.append(rmse)

    # Calculate and print average RMSE across all folds using SUM and LEN
    avg_rmse = sum(rmse_values) / len(rmse_values)
    print(f"=== Average RMSE across {len(rmse_values)} folds: {avg_rmse:.6f} ===")


if __name__ == "__main__":
    main()
