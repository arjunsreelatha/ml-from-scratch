"""Plot the derivative of the sigmoid function."""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Return the sigmoid of x."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Return the derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)


def main():
    """Generate and plot the sigmoid derivative."""
    x = np.linspace(-10, 10, 100)
    y = sigmoid_derivative(x)

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("sigmoid'(x)")
    plt.title("Derivative of Sigmoid Function")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()