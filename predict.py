import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import Net
from config import config
import data


# -------------------- 模型加载 --------------------
def load_model(model_path, device):
    """Load trained model weights."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    model = Net(4, 2, use_spectic_conv1d=True, use_spectic_transformer=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# -------------------- 通用预测函数 --------------------
def predict(model, test_loader, device):
    """Run model inference on test set (no labels)."""
    all_probs = []
    all_sequences = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            sequences, inputs = batch
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1]  # prob for class=1
            all_probs.extend(probs.cpu().numpy())
            all_sequences.extend(sequences)

    return np.array(all_sequences), np.array(all_probs)


# -------------------- Step 1: 甲基化预测 --------------------
def methylation_prediction(test_loader, output_path, device, threshold):
    print("[INFO] Step 1: Predicting methylation status using cancer_meth.pth")
    model_path = "./models/cancer_meth.pth"
    model = load_model(model_path, device)

    sequences, probs = predict(model, test_loader, device)
    preds = (probs >= threshold).astype(int)
    labels = np.where(preds == 1, "1", "0")

    df = pd.DataFrame({
        "Sequence": sequences,
        "Meth_Prob": probs,
        "Meth_Pred": labels
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Methylation predictions saved to: {output_path}")


# -------------------- Step 2: 各癌症预测 --------------------
def cancer_prediction(test_loader, output_path, device, threshold):
    print("[INFO] Step 2: Predicting cancer-specific associations")

    cancer_types = [
        "BLCA", "BRCA", "ESCA", "HNSC", "KIRC", "KIRP", "LIHC",
        "LUAD", "LUSC", "PAAD", "PCPG", "PRAD", "READ",
        "SKCM", "THYM", "UCEC"
    ]

    results = {}
    sequences = None

    for cancer in cancer_types:
        model_path = f"./models/{cancer}.pth"
        if not os.path.exists(model_path):
            print(f"[WARN] Model not found: {model_path}, skipping.")
            continue

        print(f"[INFO] Loading model: {model_path}")
        model = load_model(model_path, device)
        seqs, probs = predict(model, test_loader, device)

        if sequences is None:
            sequences = seqs

        preds = (probs >= threshold).astype(int)
        labels = np.where(preds == 1, "1", "0")

        results[f"driver_{cancer}_Prob"] = probs
        results[f"driver_{cancer}_Pred"] = labels

    # 合并结果
    df = pd.DataFrame({"Sequence": sequences})
    for k, v in results.items():
        df[k] = v

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Cancer-type predictions saved to: {output_path}")


# -------------------- Step 3: 单模型预测 --------------------
def single_model_prediction(test_loader, model_path, output_path, device, threshold):
    print(f"[INFO] Single-model prediction using {model_path}")
    model = load_model(model_path, device)

    sequences, probs = predict(model, test_loader, device)
    preds = (probs >= threshold).astype(int)
    labels = np.where(preds == 1, "1", "0")

    df = pd.DataFrame({
        "Sequence": sequences,
        "Prob": probs,
        "Pred": labels
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Single-model predictions saved to: {output_path}")


# -------------------- 主函数 --------------------
def main():
    print(f"[INFO] Using device: {config.device}")
    print(f"[INFO] Loading test data from: {config.test}")

    # 加载数据
    test_loader = data.get_test_dataloader(config.test, batch_size=config.batch_size)
    print("[INFO] Test data loaded successfully.")

    # ---------------- 参数互斥检查 ----------------
    if config.step is not None and config.model is not None:
        print("[ERROR] '-step' and '-model' cannot be used together. Choose one.")
        sys.exit(1)
    if config.step is None and config.model is None:
        print("[ERROR] You must specify either '-step' or '-model'.")
        sys.exit(1)

    # ---------------- 执行不同预测模式 ----------------
    if config.step == 1:
        methylation_prediction(test_loader, config.output, config.device, config.threshold)
    elif config.step == 2:
        cancer_prediction(test_loader, config.output, config.device, config.threshold)
    elif config.model is not None:
        single_model_prediction(
            test_loader,
            config.model,
            config.output,
            config.device,
            config.threshold
        )
    else:
        print("[ERROR] Invalid configuration. Exiting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
