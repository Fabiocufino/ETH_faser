import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def print_ascii_example(coords, labels, predictions):
    """Prints segmentation results as ASCII grid."""
    grid_true = [[" " for _ in range(6)] for _ in range(6)]
    grid_pred = [[" " for _ in range(6)] for _ in range(6)]
    
    for (x, y), label, pred in zip(coords, labels, predictions):
        grid_true[x][y] = str(label)
        grid_pred[x][y] = str(pred)

    print("True Labels:")
    print("\n".join("".join(row) for row in grid_true))
    
    print("Predicted Labels:")
    print("\n".join("".join(row) for row in grid_pred))

def compute_accuracy(labels, predictions):
    """Compute pixel-wise accuracy for segmentation."""
    correct = (labels == predictions).sum()
    total = labels.numel()
    return 100.0 * correct / total
