\
import json
import os
from typing import Any, Dict, List, Optional, Tuple

def _first_key(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None

def load_mvor_json(json_path: str) -> Dict[str, Any]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotations file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required_top = ["images", "annotations"]
    for k in required_top:
        if k not in data:
            raise KeyError(f"Missing top-level key '{k}' in {json_path}")
    return data

def index_by_image_id(images: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {img["id"]: img for img in images if "id" in img}

def _color_pref_path(images_root: str, relative_file: str) -> Optional[str]:
    """
    Prefer color path; if JSON has 'depth', try to switch to 'color' first.
    If that fails, fall back to whatever is present.
    """
    if "depth/" in relative_file:
        color_rel = relative_file.replace("depth/", "color/")
        color_abs = os.path.join(images_root, color_rel)
        if os.path.exists(color_abs):
            return color_abs
    # default
    abs_path = os.path.join(images_root, relative_file)
    if os.path.exists(abs_path):
        return abs_path
    # try color as alternative if JSON gave color but missing
    if "color/" in relative_file:
        depth_rel = relative_file.replace("color/", "depth/")
        depth_abs = os.path.join(images_root, depth_rel)
        if os.path.exists(depth_abs):
            return depth_abs
    return None

def find_image_path(images_root: str, image_dict: Dict[str, Any]) -> Optional[str]:
    """
    Resolve using images_root + file_name directly, respecting color preference.
    """
    file_name = image_dict.get("file_name") or image_dict.get("file")
    if not file_name:
        return None
    # normalize separators
    file_name = file_name.replace("\\", "/")
    # prefer color, fallback to depth
    p = _color_pref_path(images_root, file_name)
    if p:
        return p
    # last resort: recursive search by basename
    base = os.path.basename(file_name)
    for root, _, files in os.walk(images_root):
        if base in files:
            return os.path.join(root, base)
    return None

def get_optional_fields(obj: Dict[str, Any], key_candidates: List[str], default=None):
    key = _first_key(obj, key_candidates)
    return obj.get(key) if key else default

def extract_ann_fields(ann: Dict[str, Any], keyspace: Dict[str, Any]):
    head_pose = get_optional_fields(ann, keyspace["head_pose"], None)
    gaze_class = get_optional_fields(ann, keyspace["gaze_class"], None)
    kpts2d = get_optional_fields(ann, keyspace["keypoints_2d"], None)
    kpts3d = get_optional_fields(ann, keyspace["keypoints_3d"], None)
    bbox = get_optional_fields(ann, keyspace["bbox"], None)
    return head_pose, gaze_class, kpts2d, kpts3d, bbox

def get_multiview_info(dataset_json: Dict[str, Any], keyspace: Dict[str, Any]):
    mv_key = _first_key(dataset_json, keyspace["multiview_images"]) if isinstance(keyspace["multiview_images"], list) else keyspace["multiview_images"]
    cam_key = _first_key(dataset_json, keyspace["cameras_info"]) if isinstance(keyspace["cameras_info"], list) else keyspace["cameras_info"]
    mv = dataset_json.get(mv_key, [])
    cams = dataset_json.get(cam_key, {})
    return mv, cams
