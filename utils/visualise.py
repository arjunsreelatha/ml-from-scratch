"""
Utility functions for loading CSV data and plotting loss history.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import os

def load_csv(path: str, required_columns: list[str]) -> tuple[list[dict], list[tuple]]:
    """
    Load a CSV file and validate required columns.
    Returns a tuple: (valid_rows, bad_rows).
    """
    valid_rows = []
    bad_rows = []
    try:
        with open(path, "r") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                raise ValueError("CSV file is empty or has no header.")
            missing = [col for col in required_columns if col not in reader.fieldnames]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            for row_number, row in enumerate(reader, start=2):
                try:
                    cleaned = {k: (v.strip() if v is not None else "") for k, v in row.items()}
                    valid_rows.append(cleaned)
                except Exception as e:
                    bad_rows.append((row_number, str(e), row))
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except ValueError:
        raise
    return valid_rows, bad_rows

def plot_loss_history(loss_history: list[float], title: str, save_path: str | None, color: str = "blue") -> dict[str, float | int]:
    """
    Plot the loss history over epochs.
    Returns a dictionary with min loss, its iteration, and total iterations.
    """
    if not loss_history:
        raise ValueError("loss_history must not be empty")
    if any(x is None or np.isnan(x) for x in loss_history):
        raise ValueError("loss_history must not contain None or NaN values")
    min_loss = min(loss_history)
    min_idx = loss_history.index(min_loss)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color=color)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
    return {
        "min_loss": min_loss,
        "min_loss_iteration": min_idx,
        "total_iterations": len(loss_history)
    }