import math
from collections import Counter
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 100
PLOT_FIGSIZE = (8, 5)
SQUARED_ERROR_GRADIENT_FACTOR = 2.0


def entropy(labels:Sequence[int|str]) -> float:
    if labels is None:
        raise TypeError("labels must not be empty")
    try:
        hash(labels[0])
    except TypeError:
        raise TypeError("labels must be hashable")

    count = Counter(labels)
    total = len(labels)

    ent = 0.0
    for c in count.values():
        probability = c / total
        ent -= probability * math.log2(probability)

    return ent


def information_gain(parent_labels: Sequence[int|str], left_labels: Sequence[int|str], right_labels: Sequence[int|str]) -> float:
    """Calculate the information gain from splitting parent_labels into left_labels and right_labels."""
    if parent_labels is None:
        raise TypeError("parent labels must not be empty")
    if left_labels is None and right_labels is None:
        raise TypeError ("at least one of the child label sets must be non-empty")

    parent_entropy = entropy(parent_labels)
    total = len(parent_labels)

    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total

    weighted_child_entropy = 0.0

    if left_labels:
        weighted_child_entropy += left_weight * entropy(left_labels)

    if right_labels:
        weighted_child_entropy += right_weight * entropy(right_labels)

    return parent_entropy - weighted_child_entropy


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

def compute_mse(y_true: NDArray, y_pred: NDArray) -> float:
    """return the mean squared error between true and predicted values"""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def compute_descent_step(X: NDArray, y: NDArray, weight: float, bias: float, learning_rate: float) -> tuple[float, float, float]:
    """compute a single step of gradient descent and return the updated weight, bias, and loss"""
    n = len(X)

    predictions = weight * X + bias
    errors = predictions - y

    dw = (2/n)*np.dot(errors,X)
    db = (2/n)*np.sum(errors)

    weight = weight - learning_rate*dw
    bias = bias - learning_rate*db

    loss = compute_mse(y, predictions)
    return weight, bias, loss

def train_linear_model(X: NDArray, y: NDArray, weight: float = 0.0, bias: float = 0.0, learning_rate: float = 0.01, epochs: int = 100) -> tuple[float, float, list[float]]:
    """train a linear model using gradient descent and return the final weight, bias, and loss history"""
    loss_history = []
    for epoch in range(epochs):
        weight,bias,loss = compute_descent_step(X,y,weight,bias,learning_rate)
        loss_history.append(loss)

    return weight, bias, loss_history

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
    labels = [0, 0, 1, 1, 1]
    left = [0, 0]
    right = [1, 1, 1]

    print("Entropy:", entropy(labels))
    print("Information gain:", information_gain(labels, left, right))

    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([2, 3, 5, 7, 11], dtype=float)

    weights = np.array([2.0])
    bias = 0.0

    print("Predictions:", predict_linear(X, weights, bias))
    plot_regression_line(X, y, weights, bias)

    X= np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([2,4,6,8,10], dtype=float)

    final_weight,final_bias,loss_history = train_linear_model(X,y,learning_rate=0.01,epochs=100)

    print("Final weight:", final_weight)
    print("Final bias:", final_bias)
    print("Final loss:", loss_history[-1])

    plot_loss_history(loss_history)


if __name__ == "__main__":
    main()