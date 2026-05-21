import numpy as np
from numpy.typing import NDArray

from sigmoid import sigmoid
from loss_functions import bin_cross_entropy
from utils.metrics import accuracy, precision



def initialise_parameters(X: NDArray) -> tuple[NDArray, float]:
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0
    return weights, bias


def compute_logistic_gradients(X: NDArray, y: NDArray, a: NDArray) -> tuple[NDArray, float]:
    n = len(X)
    dw = (1 / n) * np.dot(X.T, (a - y)).flatten()
    db = (1 / n) * np.sum(a - y)
    return dw, db


def forward_pass(X: NDArray, w: NDArray, b: float) -> NDArray:
    z = X @ w + b
    return sigmoid(z)


def training_loop(
    X: NDArray,
    y_true: NDArray,
    learning_rate: float,
    epochs: int
) -> tuple[NDArray, float, list[float]]:
    w, b = initialise_parameters(X)
    losses = []

    for _ in range(epochs):
        a = forward_pass(X, w, b)
        dw, db = compute_logistic_gradients(X, y_true, a)
        w -= learning_rate * dw
        b -= learning_rate * db
        loss = bin_cross_entropy(y_true, a)
        losses.append(loss)

    return w, b, losses



def main():
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    w, b, losses = training_loop(X, y, learning_rate=0.1, epochs=1000)

    a = forward_pass(X, w, b)
    y_pred = (a >= 0.5).astype(int)

    print("Accuracy:", accuracy(y, y_pred))
    print("Precision:", precision(y, y_pred))
    print("Final loss:", losses[-1])


if __name__ == "__main__":
    main()