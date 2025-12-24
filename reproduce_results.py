import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix, accuracy_score
)

import data
from config import config
from model import Net


# ---------------- Helper functions ----------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


THRESHOLDS = np.linspace(0.4, 0.6, 51)[1:-1]


def matthews_corrcoef_gpu(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    tp = torch.sum((y_true == 1) & (y_pred == 1)).float()
    fp = torch.sum((y_true == 0) & (y_pred == 1)).float()
    tn = torch.sum((y_true == 0) & (y_pred == 0)).float()
    fn = torch.sum((y_true == 1) & (y_pred == 0)).float()
    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    denominator = torch.where(
        denominator == 0, torch.tensor(1e-10, device=y_true.device), denominator
    )
    return numerator / denominator


def get_best_mcc(y_true: np.ndarray, y_logits: np.ndarray):
    """Search for the optimal threshold using MCC"""
    y_true_t = torch.tensor(y_true, dtype=torch.long)
    y_probs = F.softmax(torch.tensor(y_logits, dtype=torch.float32), dim=1)

    mcc_scores = []
    best_pred = None

    for thr in THRESHOLDS:
        y_pred = (y_probs[:, 1] >= thr).long()
        mcc = matthews_corrcoef_gpu(y_true_t, y_pred)
        mcc_scores.append(mcc.item())
        if mcc.item() == max(mcc_scores):
            best_pred = y_pred

    best_idx = int(np.argmax(mcc_scores))
    best_threshold = THRESHOLDS[best_idx]
    best_mcc = mcc_scores[best_idx]

    return best_mcc, best_threshold, best_pred.cpu().numpy(), y_probs[:, 1].cpu().numpy()


# ---------------- Train & Eval function ----------------
def train_and_evaluate(data_name, seed, metric_type, device):
    """Reproduce the original training logic precisely"""
    print(f"\n[INFO] Dataset: {data_name} | Seed={seed} | MetricType={metric_type}")
    set_seed(seed)

    train_path = f"./Datasets/{data_name}/train.tsv"
    test_path = f"./Datasets/{data_name}/test.tsv"

    train_loader, test_loader = data.get_dataloader(train_path, test_path)

    model = Net(4, 2, use_spectic_conv1d=True, use_spectic_transformer=True)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_metric = -1.0
    best_threshold = None
    best_epoch = None
    best_acc = best_sn = best_sp = best_precision = best_f1 = best_auc = best_ap = best_mcc = None

    # ---------------- Training loop ----------------
    for epoch in range(200):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # ---------- validation ----------
        model.eval()
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs, _ = model(inputs)
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_outputs = np.concatenate(all_outputs)

        # Threshold search
        mcc, threshold, y_pred, prob = get_best_mcc(all_labels, all_outputs)

        # Compute evaluation metrics
        acc = accuracy_score(all_labels, y_pred)
        sn = recall_score(all_labels, y_pred)
        cm = confusion_matrix(all_labels, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = precision_score(all_labels, y_pred, zero_division=0)
        f1 = f1_score(all_labels, y_pred, zero_division=0)
        auc = roc_auc_score(all_labels, prob) if len(np.unique(all_labels)) > 1 else 0.0
        ap = average_precision_score(all_labels, prob) if len(np.unique(all_labels)) > 1 else 0.0

        current_metric = mcc if metric_type == "MCC" else ap

        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = threshold
            best_epoch = epoch + 1
            best_acc = acc
            best_sn = sn
            best_sp = sp
            best_precision = precision
            best_f1 = f1
            best_auc = auc
            best_ap = ap
            best_mcc = mcc

    # return 
    return {
        "ACC": best_acc,
        "SN": best_sn,
        "SP": best_sp,
        "Precision": best_precision,
        "F1": best_f1,
        "MCC": best_mcc,
        "AUROC": best_auc,
        "AUPRC": best_ap,
    }
'''
    return {
        "ACC": best_acc,
        "SN": best_sn,
        "SP": best_sp,
        "Precision": best_precision,
        "F1": best_f1,
        "MCC": best_mcc,
        "AUROC": best_auc,
        "AUPRC": best_ap,
        "BestMetric": best_metric,
        "BestEpoch": best_epoch,
        "Threshold": best_threshold,
    }
'''

# ---------------- Main reproduction ----------------
def main():
    device = config.decive
    os.makedirs("./reproduce_results", exist_ok=True)

    # ---------- cancer_meth ----------
    human_seed = 2150949149
    human_metrics = train_and_evaluate("cancer_meth", human_seed, metric_type="MCC", device=device)
    pd.DataFrame([human_metrics]).to_csv("./reproduce_results/cancer_meth_reproduce.csv", index=False)
    print("[INFO] Human results saved -> reproduce_results/cancer_meth_reproduce.csv")

    # ---------- Cancers ----------
    cancer_seeds = {
        "BLCA": 1376355597, "BRCA": 1376630772, "ESCA": 1376731989, "HNSC": 1376772847,
        "KIRC": 1376954675, "KIRP": 1377139197, "LIHC": 1377194207, "LUAD": 1377288340,
        "LUSC": 1377556776, "PAAD": 1378065355, "PCPG": 1378413517, "PRAD": 1378631238,
        "READ": 1378803465, "SKCM": 1379275329, "THYM": 1379777640, "UCEC": 1379893429
    }

    results = []
    for cancer, seed in cancer_seeds.items():
        metrics = train_and_evaluate(cancer, seed, metric_type="AP", device=device)
        results.append({
            "Cancer": cancer,
            "AUROC": metrics["AUROC"],
            "AUPRC": metrics["AUPRC"]
        })

    pd.DataFrame(results).to_csv("./reproduce_results/cancer_reproduce.csv", index=False)
    print("[INFO] Cancer results saved -> reproduce_results/cancer_reproduce.csv")


if __name__ == "__main__":
    main()



