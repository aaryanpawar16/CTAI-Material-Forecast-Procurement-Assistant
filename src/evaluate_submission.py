# src/evaluate_submission.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

def compute_scores(y_true_item, y_pred_item, y_true_qty, y_pred_qty):
    # classification metrics
    acc = accuracy_score(y_true_item, y_pred_item)
    f1 = f1_score(y_true_item, y_pred_item, average='macro')  # macro to treat classes evenly
    # regression MAE and normalized score
    mae = mean_absolute_error(y_true_qty, y_pred_qty)
    qty_range = float(np.nanmax(y_true_qty) - np.nanmin(y_true_qty))
    if qty_range == 0 or np.isnan(qty_range):
        reg_score = 1.0
    else:
        norm_mae = mae / qty_range
        norm_mae = max(0.0, min(1.0, norm_mae))
        reg_score = 1.0 - norm_mae
    final = 0.25 * acc + 0.25 * f1 + 0.5 * reg_score
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "mae": float(mae),
        "reg_score": float(reg_score),
        "final_score": float(final)
    }

if __name__ == "__main__":
    # sample usage: compare submission/submission.csv with a local ground truth file data/val_ground_truth.csv
    # ground truth must have columns: id, MasterItemNo, QtyShipped
    import sys, os
    gt_path = sys.argv[1] if len(sys.argv) > 1 else 'data/val_ground_truth.csv'
    sub_path = sys.argv[2] if len(sys.argv) > 2 else 'submission/submission.csv'
    if not os.path.exists(gt_path):
        print("Ground truth not found:", gt_path); sys.exit(1)
    if not os.path.exists(sub_path):
        print("Submission not found:", sub_path); sys.exit(1)
    gt = pd.read_csv(gt_path)
    sub = pd.read_csv(sub_path)
    # align by id
    merged = gt.merge(sub, on='id', suffixes=('_true','_pred'))
    res = compute_scores(merged['MasterItemNo_true'].astype(str), merged['MasterItemNo_pred'].astype(str),
                         merged['QtyShipped_true'].astype(float), merged['QtyShipped_pred'].astype(float))
    print("Evaluation results:")
    for k,v in res.items():
        print(f"{k}: {v:.6f}")
