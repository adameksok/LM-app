import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from typing import Tuple, Literal

DatasetType = Literal[
    "classification_2d",
    "regression_2d",
    "clustering_2d"
]


def generate_dataset(
    dataset_type: DatasetType,
    n_samples: int = 100,
    noise: float = 0.1,
    random_state: int = 42,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic dataset.

    Args:
        dataset_type: Type of dataset
        n_samples: Number of samples
        noise: Noise level
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters like `trend_type` for regression.

    Returns:
        (X, y) — features and labels/values
    """
    np.random.seed(random_state)

    if dataset_type == "classification_2d":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    elif dataset_type == "regression_2d":
        trend = kwargs.get("trend_type", "positive")
        X = np.random.uniform(0, 10, n_samples)
        
        if trend == "positive":
            y = 2.5 * X + 10 + np.random.normal(0, noise * 20, n_samples)
        elif trend == "negative":
            y = -2.5 * X + 40 + np.random.normal(0, noise * 20, n_samples)
        elif trend == "random":
            y = np.random.normal(20, noise * 30, n_samples)
        elif trend == "parabolic":
            # U-shape
            y = 1.5 * (X - 5)**2 + np.random.normal(0, noise * 20, n_samples)
        elif trend == "outliers":
            y = 2.5 * X + 10 + np.random.normal(0, noise * 10, n_samples)
            n_outliers = kwargs.get("n_outliers", 5)
            if n_outliers > 0:
                n_outliers = min(n_outliers, n_samples)
                idx = np.random.choice(n_samples, n_outliers, replace=False)
                y[idx] = y[idx] + np.random.choice([-1, 1], n_outliers) * np.random.uniform(50, 100, n_outliers)
        else:
            y = 2.5 * X + 10 + np.random.normal(0, noise * 20, n_samples)

        X = X.reshape(-1, 1)

    elif dataset_type == "clustering_2d":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=3,
            cluster_std=noise * 2,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return X, y


SAMPLE_DATASETS = {
    "moons": lambda: make_moons(n_samples=200, noise=0.15, random_state=42),
    "circles": lambda: make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42),
    "blobs_2": lambda: make_blobs(n_samples=200, centers=2, random_state=42),
    "blobs_3": lambda: make_blobs(n_samples=200, centers=3, random_state=42),
    "linear": lambda: generate_dataset("regression_2d", n_samples=50, noise=0.2),
}
