# src/scripts/train_attention_model.py
import argparse
import os
import random
import math

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.mvor_dataset import MVORDataset
from src.models.attention_model import AttentionPoseGazeNet
from src.utils.logging_utils import setup_logger


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.learning_rate:
        cfg["train"]["learning_rate"] = args.learning_rate
    if args.output_dir:
        cfg["project"]["output_dir"] = args.output_dir

    # ---- safe defaults for anti-collapse training ----
    cfg.setdefault("train", {})
    cfg["train"].setdefault("attention_entropy_weight", 0.05)
    cfg["train"].setdefault("attention_uniform_weight", 0.05)
    cfg["train"].setdefault("warmup_steps", 25)     # first N grad steps use mean fusion
    cfg["train"].setdefault("warn_prob", 0.03)       # 3% chance to log a low-attention warning

    # Allow fusion params to be driven by config (with defaults)
    cfg.setdefault("model", {})
    cfg["model"].setdefault("fusion_temperature", 0.7)
    cfg["model"].setdefault("fusion_min_prob", 1e-4)
    cfg["model"].setdefault("fusion_dropout", 0.1)

    os.makedirs(cfg["project"]["output_dir"], exist_ok=True)
    logger = setup_logger("train", log_file=os.path.join(cfg["project"]["output_dir"], "train.log"))
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg)
    logger.info(f"Using device: {device}")

    torch.set_num_threads(os.cpu_count())
    logger.info(f"Using {os.cpu_count()} CPU threads for dataloading and training.")

    # ---------------- Data ----------------
    train_set = MVORDataset(cfg, split="train", logger=logger)
    val_set = MVORDataset(cfg, split="val", logger=logger)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["project"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, cfg["train"]["batch_size"] // 2),
        shuffle=False,
        num_workers=cfg["project"]["num_workers"],
        pin_memory=True,
    )

    # ---------------- Model ----------------
    model = AttentionPoseGazeNet(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        att_ratio=cfg["model"]["attention_reduction"],
        gaze_classes=cfg["data"]["gaze_classes"],
        fusion=cfg["model"]["multiview_fusion"],
        dropout=cfg["model"]["dropout"],
        fusion_temperature=cfg["model"]["fusion_temperature"],
        fusion_min_prob=cfg["model"]["fusion_min_prob"],
        fusion_dropout=cfg["model"]["fusion_dropout"],
    ).to(device)

    # ---------------- Losses ----------------
    pose_criterion = nn.MSELoss()
    gaze_criterion = nn.CrossEntropyLoss()

    # ---------------- Optimizer ----------------
    opt = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"]["betas"]),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg["project"]["mixed_precision"] and device.type == "cuda"
    )

    best_val_metric = float("inf")
    best_path = os.path.join(cfg["project"]["output_dir"], "best_attention_model.pth")

    # global step for warmup & ramps
    global_step = 0

    def cosine_ramp(t: float) -> float:
        # t in [0, 1] -> ramp from 1 to 0 using cosine
        t = max(0.0, min(1.0, t))
        return 0.5 * (1.0 + math.cos(t * math.pi))

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        # ---------------- Train ----------------
        model.train()
        tr_pose_loss = tr_gaze_loss = tr_total = 0.0

        for i, (x, y_pose, y_gaze, meta) in enumerate(train_loader):
            x = x.to(device)  # (B, V, 3, H, W)
            y_pose = y_pose.to(device)
            y_gaze = y_gaze.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                # warm-up: force mean fusion to avoid early attention collapse
                force_mean = global_step < cfg["train"]["warmup_steps"]
                pose_pred, gaze_logits, att = model(x, force_mean_fusion=force_mean)

                loss_pose = pose_criterion(pose_pred, y_pose)
                loss_gaze = gaze_criterion(gaze_logits, y_gaze)
                loss = loss_pose + loss_gaze

                # anti-collapse regularization (only when attention is active)
                if (att is not None) and (not force_mean):
                    # encourage spread
                    att_entropy = -torch.sum(att * torch.log(att + 1e-8), dim=1).mean()
                    # encourage balance (close to uniform)
                    uniform = torch.full_like(att, 1.0 / att.shape[1])
                    uniform_loss = torch.mean(torch.abs(att - uniform))

                    # cosine ramp: strong early, gentler later
                    ramp = cosine_ramp(global_step / max(1, cfg["train"]["warmup_steps"] * 4))
                    w_ent = float(cfg["train"]["attention_entropy_weight"]) * float(ramp)
                    w_uni = float(cfg["train"]["attention_uniform_weight"]) * float(ramp)

                    loss = loss + w_ent * att_entropy + w_uni * uniform_loss

            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip_norm"] and cfg["train"]["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            scaler.step(opt)
            scaler.update()

            tr_pose_loss += loss_pose.item()
            tr_gaze_loss += loss_gaze.item()
            tr_total += loss.item()

            # Throttled attention monitoring
            if att is not None and not force_mean:
                att_min = float(att.min().item())
                # log with probability to avoid spam
                if att_min < cfg["eval"]["attention_alert_threshold"]:
                    if random.random() < float(cfg["train"]["warn_prob"]):
                        logger.warning(
                            f"Low attention weight detected: min={att_min:.3f} "
                            f"(batch {i}, step {global_step})"
                        )
                # periodic mean attn
                if (i + 1) % 20 == 0:
                    mean_attn = att.mean(dim=0).detach().cpu().numpy()
                    logger.info(f"mean attention weights per view: {mean_attn}")

            if (i + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch} [{i+1}/{len(train_loader)}] "
                    f"total={loss.item():.4f} pose={loss_pose.item():.4f} gaze={loss_gaze.item():.4f}"
                )

            global_step += 1

        ntr = max(1, len(train_loader))
        logger.info(
            f"[Train] Epoch {epoch}: total={tr_total/ntr:.4f} "
            f"pose={tr_pose_loss/ntr:.4f} gaze={tr_gaze_loss/ntr:.4f}"
        )

        # ---------------- Validate ----------------
        if epoch % cfg["train"]["val_interval"] == 0:
            model.eval()
            vl_pose = vl_gaze = vl_total = 0.0
            with torch.no_grad():
                for x, y_pose, y_gaze, meta in val_loader:
                    x = x.to(device)
                    y_pose = y_pose.to(device)
                    y_gaze = y_gaze.to(device)
                    # no warmup at val time
                    pose_pred, gaze_logits, _ = model(x, force_mean_fusion=False)
                    loss_pose = pose_criterion(pose_pred, y_pose)
                    loss_gaze = gaze_criterion(gaze_logits, y_gaze)
                    loss = loss_pose + loss_gaze
                    vl_pose += loss_pose.item()
                    vl_gaze += loss_gaze.item()
                    vl_total += loss.item()
            nvl = max(1, len(val_loader))
            val_total = vl_total / nvl
            logger.info(
                f"[Val]   Epoch {epoch}: total={val_total:.4f} "
                f"pose={vl_pose/nvl:.4f} gaze={vl_gaze/nvl:.4f}"
            )

            # Save best
            if val_total < best_val_metric:
                best_val_metric = val_total
                torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)
                logger.info(f"Saved best model to {best_path} (val_total={best_val_metric:.4f})")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
