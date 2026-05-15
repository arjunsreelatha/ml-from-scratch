import math
from collections import Counter
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from foundations.loss_functions import compute_mse


DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 100
PLOT_FIGSIZE = (8, 5)


def predict_linear(X:NDArray, weights:NDArray, bias:float) -> NDArray:
    """Make predictions using a linear model. y =Xw+b"""
    shape1 = X.shape
    shape2 = weights.shape
    if shape1[1] != shape2[0]:
        raise ValueError(f"X has shape {shape1} but weights has shape {shape2}, incompatible for multiplication")
    return np.dot(X, weights) + bias


def plot_regression_line(X:NDArray, y:NDArray, weights:NDArray, bias:float  ) -> None:
    """plot actual data points and the regression line defined by weights an bias"""
    predictions = predict_linear(X, weights, bias)

    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], y, color="blue", label="Actual data")
    plt.plot(X[:, 0], predictions, color="red", label="Regression line")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_linear_gradients(X: NDArray, y: NDArray, weights: NDArray, bias: float) -> tuple[NDArray, float]:
    """compute the gradients of the loss with respect to weights and bias for a linear model"""
    n = len(X)

    predictions = predict_linear(X, weights, bias)
    errors = predictions - y

    dw = (2/n)*np.dot(X.T, errors)
    db = (2/n)*np.sum(errors)

    return dw, db,predictions

def compute_descent_step(X: NDArray, y: NDArray, weights: NDArray, bias: float, learning_rate: float) -> tuple[NDArray, float, float]:
    """compute a single step of gradient descent and return the updated weights, bias, and loss"""

    dw, db, predictions = computute_linear_gradients(X, y, weights, bias)    

    weights = weights - learning_rate*dw
    bias = bias - learning_rate*db
    loss = compute_mse(y, predictions)
    return weights, bias, loss

def train_linear_model(X: NDArray, y: NDArray, weights: NDArray = None, bias: float = 0.0, learning_rate: float = 0.01, epochs: int = 100) -> tuple[NDArray, float, list[float]]:
    """train a linear model using gradient descent and return the final weights, bias, and loss history"""
    if weights is None:
        weights = np.zeros(X.shape[1])

    loss_history = []
    for epoch in range(epochs):
        weights,bias,loss = compute_descent_step(X,y,weights,bias,learning_rate)
        loss_history.append(loss)

    return weights, bias, loss_history

def plot_loss_history(loss_history: list[float]) -> None:
    """Plot the loss history over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.show()

def main():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([2, 3, 5, 7, 11], dtype=float)

    weights = np.array([2.0])
    bias = 0.0

    print("Predictions:", predict_linear(X, weights, bias))
    plot_regression_line(X, y, weights, bias)

    X= np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([2,4,6,8,10], dtype=float)

    final_weights,final_bias,loss_history = train_linear_model(X,y,learning_rate=0.01,epochs=100)

    print("Final weights:", final_weights)
    print("Final bias:", final_bias)
    print("Final loss:", loss_history[-1])

    plot_loss_history(loss_history)


if __name__ == "__main__":
    main()