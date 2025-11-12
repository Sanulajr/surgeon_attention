import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
import seaborn as sns

from src.data.mvor_dataset import MVORDataset
from src.models.attention_model import AttentionPoseGazeNet
from src.utils.logging_utils import setup_logger

# ------------------- Helpers -------------------
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_device(cfg):
    if cfg["project"]["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ------------------- Main -------------------
def main():
    cfg_path = "src/config.yaml"
    model_path = "outputs/best_attention_model.pth"
    out_dir = "outputs/analysis"
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(cfg_path)
    device = get_device(cfg)
    logger = setup_logger("analysis", log_file=os.path.join(out_dir, "analysis.log"))
    logger.info(f"Using device: {device}")

    # Dataset
    val_set = MVORDataset(cfg, split="val", logger=logger)
    val_loader = DataLoader(val_set, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                            num_workers=cfg["project"]["num_workers"], pin_memory=True)

    # Model
    model = AttentionPoseGazeNet(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        att_ratio=cfg["model"]["attention_reduction"],
        gaze_classes=cfg["data"]["gaze_classes"],
        fusion=cfg["model"]["multiview_fusion"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ------------------- Evaluation Loop -------------------
    all_true_gaze, all_pred_gaze = [], []
    yaw_errors, pitch_errors = [], []

    with torch.no_grad():
        for x, y_pose, y_gaze, meta in val_loader:
            x = x.to(device)
            y_pose = y_pose.to(device)
            y_gaze = y_gaze.to(device)

            pose_pred, gaze_logits, att = model(x)

            pred_cls = torch.argmax(gaze_logits, dim=1)
            all_true_gaze.extend(y_gaze.cpu().numpy().tolist())
            all_pred_gaze.extend(pred_cls.cpu().numpy().tolist())

            pose_err = torch.abs(pose_pred - y_pose).cpu().numpy()
            yaw_errors.extend(pose_err[:, 0])
            pitch_errors.extend(pose_err[:, 1])

    all_true_gaze = np.array(all_true_gaze)
    all_pred_gaze = np.array(all_pred_gaze)
    yaw_errors = np.array(yaw_errors)
    pitch_errors = np.array(pitch_errors)

    # ------------------- Reports -------------------
    class_names = ["Left", "Front-Left", "Front", "Front-Right", "Right"]

    cm = confusion_matrix(all_true_gaze, all_pred_gaze, labels=range(len(class_names)))
    acc_per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    logger.info("=== Classification Report ===")

    unique_true = np.unique(all_true_gaze)
    unique_pred = np.unique(all_pred_gaze)
    logger.info(f"Unique true gaze classes: {unique_true.tolist()}")
    logger.info(f"Unique predicted gaze classes: {unique_pred.tolist()}")

    try:
        logger.info(classification_report(all_true_gaze, all_pred_gaze, target_names=class_names, digits=3, labels=list(range(len(class_names)))))
    except ValueError as e:
        logger.warning(f"Classification report could not be computed normally: {e}")
        # Fallback basic accuracy
        acc = (all_true_gaze == all_pred_gaze).mean() * 100
        logger.info(f"Fallback accuracy: {acc:.2f}%")

        logger.info(f"Overall gaze accuracy: {(all_true_gaze == all_pred_gaze).mean()*100:.2f}%")
        logger.info("Per-class accuracy:")
        for name, acc in zip(class_names, acc_per_class):
            logger.info(f"  {name}: {acc*100:.2f}%")

    # ------------------- Plots -------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (Gaze Classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(yaw_errors, bins=30, alpha=0.6, label="Yaw Error")
    plt.hist(pitch_errors, bins=30, alpha=0.6, label="Pitch Error")
    plt.legend()
    plt.xlabel("Absolute Error (degrees)")
    plt.ylabel("Frequency")
    plt.title("Pose Error Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pose_error_hist.png"))
    plt.close()

    logger.info(f"Saved analysis to: {out_dir}")

if __name__ == "__main__":
    main()
