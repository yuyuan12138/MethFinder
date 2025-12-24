import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix, accuracy_score
)

import data
from model import Net
from config import config


# ----------------------- Utils: Seed -----------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------- Init seed -----------------------
if getattr(config, "seed", None) is None:
    seed = int(time.time() * 1000) % (2**32 - 1)
else:
    seed = config.seed

set_seed(seed)
print(f"[INFO] Using random seed: {seed}")

os.makedirs("./seeds", exist_ok=True)
with open(f"./seeds/{config.data_name}_{config.loop}.txt", "w") as f:
    f.write(str(seed))


# ----------------------- Data -----------------------
print(f"[INFO] Loading dataset: {config.data_name}")
train_loader, test_loader = data.get_dataloader(
    train_path=os.path.join(config.data_path, config.data_name, "train.tsv"),
    test_path=os.path.join(config.data_path, config.data_name, "test.tsv"),
)
print("[INFO] Data loaded successfully.")


# ----------------------- Model -----------------------
model = Net(4, 2, use_spectic_conv1d=True, use_spectic_transformer=True)
model.to(config.device)
print(f"[INFO] Model created on device: {config.device}")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)


# ----------------------- Metric Functions -----------------------
def matthews_corrcoef_gpu(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    tp = torch.sum((y_true == 1) & (y_pred == 1)).float()
    fp = torch.sum((y_true == 0) & (y_pred == 1)).float()
    tn = torch.sum((y_true == 0) & (y_pred == 0)).float()
    fn = torch.sum((y_true == 1) & (y_pred == 0)).float()
    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    denominator = torch.where(denominator == 0, torch.tensor(1e-10, device=y_true.device), denominator)
    return numerator / denominator


THRESHOLDS = np.linspace(0.0, 1.0, 201)[1:-1]


def get_best_threshold(y_true: np.ndarray, y_logits: np.ndarray, metric_type="MCC"):
    """
    自动根据 metric_type 选择最佳阈值。
    支持: MCC, F1, Precision, Recall, AP, AUC
    """
    y_true_t = torch.tensor(y_true, dtype=torch.long)
    y_probs = F.softmax(torch.tensor(y_logits, dtype=torch.float32), dim=1)[:, 1]

    best_thr = 0.5
    best_score = -1
    best_pred = None

    for thr in THRESHOLDS:
        y_pred = (y_probs >= thr).long()
        y_true_np = y_true_t.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        if metric_type == "MCC":
            score = matthews_corrcoef_gpu(y_true_t, y_pred).item()
        elif metric_type == "F1":
            score = f1_score(y_true_np, y_pred_np, zero_division=0)
        elif metric_type == "Precision":
            score = precision_score(y_true_np, y_pred_np, zero_division=0)
        elif metric_type == "Recall":
            score = recall_score(y_true_np, y_pred_np, zero_division=0)
        elif metric_type == "AP":
            score = average_precision_score(y_true_np, y_probs.cpu().numpy())
            return score, 0.5, (y_probs >= 0.5).long().cpu().numpy(), y_probs.cpu().numpy()
        elif metric_type == "AUC":
            score = roc_auc_score(y_true_np, y_probs.cpu().numpy())
            return score, 0.5, (y_probs >= 0.5).long().cpu().numpy(), y_probs.cpu().numpy()
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")

        if score > best_score:
            best_score = score
            best_thr = thr
            best_pred = y_pred

    return best_score, best_thr, best_pred.cpu().numpy(), y_probs.cpu().numpy()


# ----------------------- Training Loop -----------------------
print("[INFO] Start training...")
print(f"[INFO] Threshold selection metric: {config.th_metric}")

best_metric = -1.0
best_threshold = None
best_epoch = None
best_acc = best_sn = best_sp = best_precision = best_f1 = best_auc = best_ap = None
best_mcc = None
temp_model_path = None

for epoch in range(config.epochs):
    model.train()
    print(f"\n[Epoch {epoch + 1}/{config.epochs}] -------------------------")

    for step, (inputs, labels) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (step % 10 == 0) or (step == len(train_loader)):
            print(f"  Batch [{step}/{len(train_loader)}] | Loss = {loss.item():.4f}")

    # ----------------------- Validation -----------------------
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs, _ = model(inputs)
            val_loss += criterion(outputs, labels).item()
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    # 选择最优阈值
    best_score, threshold, y_pred, prob = get_best_threshold(
        all_labels, all_outputs, config.th_metric
    )

    # 计算指标
    acc = accuracy_score(all_labels, y_pred)
    sn = recall_score(all_labels, y_pred)
    cm = confusion_matrix(all_labels, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(all_labels, y_pred, zero_division=0)
    f1 = f1_score(all_labels, y_pred, zero_division=0)
    auc = roc_auc_score(all_labels, prob) if len(np.unique(all_labels)) > 1 else 0.0
    ap = average_precision_score(all_labels, prob) if len(np.unique(all_labels)) > 1 else 0.0
    mcc = matthews_corrcoef_gpu(torch.tensor(all_labels), torch.tensor(y_pred)).item()

    current_metric = best_score

    # ----------------------- Save Best -----------------------
    if current_metric > best_metric:
        best_metric = current_metric
        best_threshold = threshold
        best_epoch = epoch + 1

        os.makedirs("./models", exist_ok=True)
        temp_model_path = f"./models/{config.data_name}_{config.loop}_best_temp.pth"
        torch.save(model.state_dict(), temp_model_path)

        best_acc = acc
        best_sn = sn
        best_sp = sp
        best_precision = precision
        best_f1 = f1
        best_auc = auc
        best_ap = ap
        best_mcc = mcc

        print(
            f"\n[INFO] New best model saved | {config.th_metric}={best_metric:.4f} | "
            f"Threshold={best_threshold:.3f} | Epoch={best_epoch}"
        )

    # ----------------------- Epoch Summary -----------------------
    print(
        f"Val Loss: {val_loss / len(test_loader):.4f} | "
        f"MCC: {mcc:.4f} | AP: {ap:.4f} | AUC: {auc:.4f} | "
        f"Best_{config.th_metric}: {best_metric:.4f} | "
        f"Best_Threshold={best_threshold:.3f}"
    )


# ----------------------- Finalize -----------------------
if temp_model_path is not None and os.path.exists(temp_model_path):
    final_model_path = f"./models/{config.data_name}_{config.loop}.pth"
    os.rename(temp_model_path, final_model_path)
    print(f"\n[INFO] Training complete. Final model saved as:\n  {final_model_path}")
else:
    print("\n[WARN] No model was saved during training (no improvement found). Skipping model rename.")
