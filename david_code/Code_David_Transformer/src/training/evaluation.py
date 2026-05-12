"""
Model evaluation utilities.
"""

import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import logging


def eval_best_model(best_model, data_loaders, Activities, save_name, mode='test',
                    device=None, dir_out=None, timestamp=None, return_predictions=False):
    """
    Evaluate best model and generate confusion matrix.

    Args:
        best_model: Trained model
        data_loaders: Dictionary with data loaders
        Activities: List of class names
        save_name: Name for saving outputs
        mode: Which data loader to use ('test', 'source', 'target')
        device: torch device
        dir_out: Output directory
        timestamp: Timestamp string
        return_predictions: If True, return (accuracy, y_true, y_pred) instead of just accuracy

    Returns:
        accuracy (float), or (accuracy, y_true, y_pred) when return_predictions=True
    """
    logging.info("-------Curve-level performance evaluation-------")
    best_model.eval()
    best_model.to(device)
    pred_list  = torch.tensor([]).to(device)
    label_list = torch.tensor([]).to(device)

    with torch.no_grad():
        for data, label in data_loaders[mode]:
            data, label = data.to(device), label.to(device)
            features, output = best_model(data)

            pred  = torch.flatten(output.data.max(1, keepdim=True)[1])
            label = torch.flatten(label)

            pred_list  = torch.cat((pred_list,  pred),  0)
            label_list = torch.cat((label_list, label), 0)

    y_true = label_list.cpu().numpy()
    y_pred = pred_list.cpu().numpy()

    # Classification report
    report = classification_report(y_true, y_pred, digits=5, target_names=Activities)
    logging.info("\n" + report)

    # Per-fold confusion matrix
    _save_confusion_matrix(y_true, y_pred, Activities,
                           title=f'Confusion Matrix - {save_name}',
                           save_path=os.path.join(dir_out, f"confusion_matrix_{save_name}_{timestamp}.png")
                           if dir_out and timestamp else None)

    accuracy = 100.0 * (y_pred == y_true).sum() / len(y_true)
    if return_predictions:
        return accuracy, y_true, y_pred
    return accuracy


def _save_confusion_matrix(y_true, y_pred, Activities, title, save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Activities)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Confusion matrix saved to: {save_path}")
    plt.close()


def save_aggregated_confusion_matrix(all_y_true, all_y_pred, Activities, dir_out, timestamp):
    """
    Plot and save one confusion matrix aggregated across all CV folds.
    Each sample appears exactly once (its held-out fold prediction).
    """
    import numpy as np
    logging.info("-------Aggregated CV confusion matrix-------")

    report = classification_report(all_y_true, all_y_pred, digits=5, target_names=Activities)
    logging.info("\n" + report)

    accuracy = 100.0 * (all_y_pred == all_y_true).sum() / len(all_y_true)
    logging.info(f"Aggregated accuracy over all folds: {accuracy:.2f}%")

    save_path = os.path.join(dir_out, f"confusion_matrix_CV_aggregated_{timestamp}.png")
    _save_confusion_matrix(
        all_y_true, all_y_pred, Activities,
        title=f'Aggregated CV Confusion Matrix  (all folds, n={len(all_y_true)},  acc={accuracy:.1f}%)',
        save_path=save_path,
    )
    return accuracy

