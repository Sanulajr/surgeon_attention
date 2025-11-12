import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from arrow import get
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.utils.mvor_parser import (
    load_mvor_json,
    index_by_image_id,
    find_image_path,
    extract_ann_fields,
    get_multiview_info,
)


@dataclass
class Sample:
    image_paths: List[str]               # One or multiple view image paths
    bbox: Optional[Tuple[float, float, float, float]]
    yaw: float
    pitch: float
    gaze_class: int
    image_id: int
    person_id: Optional[int]
    rel_path: str                        # original JSON file_name


def _infer_pose_from_kpts(kpts: List[float]) -> Tuple[float, float]:
    try:
        pts = np.array(kpts, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] <= 2:
            return 0.0, 0.0
        # Heuristic: use first three points as nose, left_eye, right_eye
        nose = pts[0][:2]
        le = pts[1][:2]
        re = pts[2][:2]
        mid_eye = (le + re) / 2.0
        inter_ocular = max(np.linalg.norm(le - re), 1.0)
        yaw = 90.0 * float((nose[0] - mid_eye[0]) / inter_ocular)
        pitch = 90.0 * float((mid_eye[1] - nose[1]) / inter_ocular)
        yaw = float(np.clip(yaw, -90.0, 90.0))
        pitch = float(np.clip(pitch, -90.0, 90.0))
        return yaw, pitch
    except Exception:
        return 0.0, 0.0


def _yaw_pitch_to_gaze_class(yaw: float, pitch: float, n_classes: int = 5) -> int:
    if n_classes < 5:
        n_classes = 5
    if yaw < -30:
        return 0
    elif yaw < -10:
        return 1
    elif yaw <= 10 and abs(pitch) <= 45:
        return 2
    elif yaw <= 30:
        return 3
    else:
        return 4


class MVORDataset(Dataset):
    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str = "train",
        transform: Optional[Any] = None,
        logger: Optional[Any] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.logger = logger
        self.images_root = cfg["data"]["images_root"]
        self.annotations_file = cfg["data"]["annotations_file"]
        self.use_multiview = bool(cfg["data"].get("use_multiview", True))
        self.use_3d = bool(cfg["data"].get("use_3d", True))
        self.use_rgb_only = bool(cfg["data"].get("use_rgb_only", True))
        self.train_val_split_ratio = float(cfg["data"].get("train_val_split", 0.85))
        self.gaze_classes = int(cfg["data"].get("gaze_classes", 5))
        self.keyspace = cfg["data"]["json_keys"]
        self._load()

    def _load(self):
        print("🔍 Starting dataset load...")
        data = load_mvor_json(self.annotations_file)
        images = data["images"]
        anns = data["annotations"]
        print(f"📸 Total images in JSON: {len(images)}")
        print(f"🧍 Total annotations in JSON: {len(anns)}")

        self.img_index = index_by_image_id(images)
        mv, cams = get_multiview_info(data, self.keyspace)
        self.multiview_index = {}

        # --- Build multiview index safely ---
        if isinstance(mv, list):
            for entry in mv:
                raw_ids = entry.get("image_ids") or entry.get("images") or []
                ids = []
                for item in raw_ids:
                    if isinstance(item, dict):
                        if "id" in item:
                            ids.append(item["id"])
                        elif "image_id" in item:
                            ids.append(item["image_id"])
                    elif isinstance(item, int):
                        ids.append(item)
                ids = [i for i in ids if isinstance(i, int)]
                for iid in ids:
                    self.multiview_index[iid] = [j for j in ids if j != iid]

        # --- Main annotation loop ---
        all_samples: List[Sample] = []
        missing_images = 0
        accepted = 0
        skipped_no_image = 0
        skipped_missing_labels = 0
        infer_pose_if_missing = True

        for ann in anns:
            image_id = ann.get("image_id") or ann.get("img_id") or ann.get("imageId")
            if image_id is None or image_id not in self.img_index:
                continue
            img_info = self.img_index[image_id]

            img_path = find_image_path(self.images_root, img_info)
            if img_path is None or (not os.path.exists(img_path)):
                missing_images += 1
                skipped_no_image += 1
                continue

            image_paths = [img_path]
            if self.use_multiview and image_id in self.multiview_index:
                for mv_id in self.multiview_index[image_id]:
                    if mv_id in self.img_index:
                        mv_info = self.img_index[mv_id]
                        mv_path = find_image_path(self.images_root, mv_info)
                        if mv_path and os.path.exists(mv_path):
                            image_paths.append(mv_path)

            head_pose, gaze_class, kpts2d, kpts3d, bbox = extract_ann_fields(ann, self.keyspace)
            yaw, pitch = 0.0, 0.0
            if isinstance(head_pose, dict):
                yaw = float(head_pose.get("yaw", 0.0))
                pitch = float(head_pose.get("pitch", 0.0))
            elif isinstance(head_pose, (list, tuple)) and len(head_pose) >= 2:
                yaw, pitch = float(head_pose[0]), float(head_pose[1])
            elif infer_pose_if_missing and kpts2d is not None:
                yaw, pitch = _infer_pose_from_kpts(kpts2d)

            if gaze_class is None:
                raw_label = ann.get("category_id") or ann.get("gaze_label") or ann.get("gaze_direction")

                if isinstance(raw_label, str):
                    raw_label = raw_label.strip().lower()
                    label_map = {
                        "left": 0,
                        "fl": 1, "front-left": 1, "frontleft": 1,
                        "front": 2,
                        "fr": 3, "front-right": 3, "frontright": 3,
                        "right": 4,
                    }
                    gaze_class = label_map.get(raw_label, 2)
                else:
                    try:
                        gaze_class = int(raw_label)
                        if gaze_class not in range(self.gaze_classes):
                            gaze_class = gaze_class - 1 if 1 <= gaze_class <= self.gaze_classes else 2
                    except Exception:
                        gaze_class =_yaw_pitch_to_gaze_class(yaw, pitch, self.gaze_classes)

            if gaze_class is None:
                skipped_missing_labels += 1
                continue

            bbox_tuple = None
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                bbox_tuple = (
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                )

            person_id = ann.get("id") or ann.get("person_id") or ann.get("track_id")
            rel_path = img_info.get("file_name") or img_info.get("file") or ""

            all_samples.append(
                Sample(
                    image_paths=image_paths,
                    bbox=bbox_tuple,
                    yaw=float(yaw),
                    pitch=float(pitch),
                    gaze_class=int(gaze_class),
                    image_id=int(image_id),
                    person_id=int(person_id) if person_id is not None else None,
                    rel_path=rel_path,
                )
            )
            accepted += 1

        print("📊 Summary:")
        print(f"   ✅ Accepted samples: {accepted}")
        print(f"   🚫 Missing images: {missing_images}")
        print(f"   🚫 Skipped (no image): {skipped_no_image}")
        print(f"   🚫 Skipped (missing labels): {skipped_missing_labels}")

        if self.logger:
            self.logger.info(
                f"Loaded {len(all_samples)} samples ({missing_images} missing images skipped)."
            )

        # -------- Day-wise split: train on day1-3, val on day4 (if available) --------
        train_list, val_list = [], []
        for s in all_samples:
            rp = s.rel_path.replace("\\", "/")
            if "/day4/" in rp or rp.startswith("day4/"):
                val_list.append(s)
            else:
                train_list.append(s)

        if self.split == "train":
            self.samples = (
                train_list
                if len(val_list) > 0
                else all_samples[: int(len(all_samples) * self.train_val_split_ratio)]
            )
        else:
            self.samples = (
                val_list
                if len(val_list) > 0
                else all_samples[int(len(all_samples) * self.train_val_split_ratio) :]
            )

        print(f"📦 Final split '{self.split}': {len(self.samples)} samples")

        if self.logger:
            self.logger.info(
                f"{self.split}: {len(self.samples)} samples (day-wise split {'enabled' if len(val_list)>0 else 'fallback random'})."
            )

        # -------- Transforms (RGB only pipeline) --------
        size = tuple(self.cfg["data"]["input_size"])
        aug = self.cfg["data"]["augmentation"]
        if self.split == "train":
            self.transform = self.transform or T.Compose(
                [
                    T.Resize(size),
                    T.ColorJitter(*aug["color_jitter"]),
                    T.RandomHorizontalFlip(p=aug["horizontal_flip_prob"]),
                    T.RandomRotation(degrees=aug["rotation_degrees"]),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = self.transform or T.Compose(
                [
                    T.Resize(size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        with Image.open(path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            return im.copy()

    def __getitem__(self, idx: int):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                s = self.samples[idx]

                # --- Load and check images ---
                imgs = []
                for p in s.image_paths:
                    try:
                        img = self._load_image(p)
                        imgs.append(self.transform(img))
                    except Exception as e:
                        print(f"[WARN] Failed to load image {p}: {e}")
                        continue

                if not imgs:
                    raise ValueError(f"No valid images for sample {idx}")

                x = torch.stack(imgs, dim=0)  # (V, 3, H, W)

                # --- Check labels ---
                if s.yaw is None or s.pitch is None:
                    raise ValueError("Pose label missing")
                if s.gaze_class is None:
                    raise ValueError("Gaze class label missing")

                y_pose = torch.tensor([float(s.yaw), float(s.pitch)], dtype=torch.float32)
                y_gaze = torch.tensor(int(s.gaze_class), dtype=torch.long)

                meta = {
                    "image_paths": s.image_paths or [],
                    "bbox": s.bbox if s.bbox is not None else (0.0, 0.0, 0.0, 0.0),
                    "image_id": int(s.image_id) if s.image_id is not None else -1,
                    "person_id": int(s.person_id) if s.person_id is not None else -1,
                    "rel_path": s.rel_path or "",
                }

                return x, y_pose, y_gaze, meta

            except Exception as e:
                print(f"[WARN] Error in sample {idx} (attempt {attempt+1}/{max_retries}): {e}")
                idx = random.randint(0, len(self.samples) - 1)  # pick a new random sample

        # If all retries failed, return a dummy tensor (safe fallback)
        print(f"[ERROR] All retries failed for sample {idx}, returning blank tensor.")
        dummy_x = torch.zeros((1, 3, *self.cfg["data"]["input_size"]), dtype=torch.float32)
        dummy_pose = torch.zeros(2, dtype=torch.float32)
        dummy_gaze = torch.tensor(0, dtype=torch.long)
        dummy_meta = {"image_paths": [], "bbox": (0.0, 0.0, 0.0, 0.0), "image_id": -1, "person_id": -1, "rel_path": ""}
        return dummy_x, dummy_pose, dummy_gaze, dummy_meta
