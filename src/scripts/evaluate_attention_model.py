\
import argparse
import os

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2

from src.data.mvor_dataset import MVORDataset
from src.models.attention_model import AttentionPoseGazeNet
from src.utils.logging_utils import setup_logger
from src.utils.visualization import visualize_predictions, save_image

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_device(cfg):
    if cfg["project"]["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    parser.add_argument("--model_path", type=str, default="outputs/best_attention_model.pth")
    parser.add_argument("--out_dir", type=str, default="outputs/eval")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config(args.config)
    device = get_device(cfg)
    logger = setup_logger("eval", log_file=os.path.join(cfg["project"]["output_dir"], "eval.log"))
    logger.info(f"Using device: {device}")

    # Data
    val_set = MVORDataset(cfg, split="val", logger=logger)
    val_loader = DataLoader(val_set, batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=cfg["project"]["num_workers"], pin_memory=True)

    # Model
    model = AttentionPoseGazeNet(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        att_ratio=cfg["model"]["attention_reduction"],
        gaze_classes=cfg["data"]["gaze_classes"],
        fusion=cfg["model"]["multiview_fusion"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Metrics
    pose_errors = []
    gaze_correct = 0
    gaze_total = 0

    with torch.no_grad():
        for bidx, (x, y_pose, y_gaze, meta) in enumerate(val_loader):
            x = x.to(device)
            y_pose = y_pose.to(device)
            y_gaze = y_gaze.to(device)

            pose_pred, gaze_logits, att = model(x)
            # Pose MAE per angle
            mae = torch.abs(pose_pred - y_pose).mean(dim=0)  # [yaw_mae, pitch_mae]
            pose_errors.append(mae.detach().cpu().numpy())

            # Gaze accuracy
            pred_cls = torch.argmax(gaze_logits, dim=1)
            gaze_correct += int((pred_cls == y_gaze).sum().item())
            gaze_total += int(y_gaze.numel())

            if args.visualize and bidx < 10:
                # Save a grid of first N samples
                B = x.shape[0]


                for i in range(min(B, cfg["eval"]["visualize_samples"])):
                    if "image_paths" not in meta or i >= len(meta["image_paths"]):
                        continue

                    img_item = meta["image_paths"][i]
                    if isinstance(img_item, (list, tuple)):
                        img_item = img_item[0] if len(img_item) > 0 else None
                    img_path = str(img_item) if img_item is not None else None

                    if not img_path or not os.path.exists(img_path):
                        continue
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Draw predictions
                    pred_yaw = float(pose_pred[i,0].item())
                    pred_pitch = float(pose_pred[i,1].item())
                    pred_label = int(pred_cls[i].item())
                    vis = visualize_predictions(img, meta["bbox"][i] if "bbox" in meta and meta["bbox"][i] is not None else None, pred_yaw, pred_pitch, pred_label, class_names=["Left","FL","Front","FR","Right"])
                    out_path = os.path.join(args.out_dir, f"vis_{bidx}_{i}.png")
                    save_image(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    pose_errors = np.stack(pose_errors, axis=0) if len(pose_errors) else np.zeros((1,2), dtype=np.float32)
    yaw_mae, pitch_mae = pose_errors.mean(axis=0).tolist()
    gaze_acc = gaze_correct / max(1, gaze_total)

    logger.info(f"Validation MAE: yaw={yaw_mae:.3f}, pitch={pitch_mae:.3f}")
    logger.info(f"Validation gaze accuracy: {gaze_acc*100:.2f}%")

    # Print also to stdout for convenience
    print(f"MAE yaw={yaw_mae:.3f}, pitch={pitch_mae:.3f}; gaze_acc={gaze_acc*100:.2f}%")

if __name__ == "__main__":
    main()
