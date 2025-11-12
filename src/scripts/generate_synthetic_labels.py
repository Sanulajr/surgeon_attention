import json
import os
import random
import math

def simulate_yaw_from_cam(cam_id):
    """Rough yaw estimate based on camera position."""
    base_yaw =  {
        1: -45,
        2: -15,
        3: 15,
        4: 45
    }.get(cam_id, 0)
    yaw = base_yaw + random.uniform(-20, 20)
    return yaw
def simulate_pitch():
    """Simulate small random pitch deviations."""
    return random.uniform(-10, 10)

def yaw_to_class(yaw):
    """Map yaw angle to coarse gaze class."""
    if yaw < -30:
        return 0  # Left
    elif -30 <= yaw < -10:
        return 1  # Front-left
    elif -10 <= yaw <= 10:
        return 2  # Front
    elif 10 < yaw <= 30:
        return 3  # Front-right
    else:
        return 4  # Right

def main():
    src_path = "MVOR/annotations/camma_mvor_2018.json"
    out_path = "MVOR/annotations/camma_mvor_2018_synthetic.json"

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ✅ Access the correct list
    if "multiview_images" not in data:
        raise ValueError("JSON missing key 'multiview_images' — cannot find grouped frames.")

    groups = data["multiview_images"]
    new_annotations = []
    ann_id = 0

    for group in groups:
        if "images" not in group:
            continue

        for img in group["images"]:
            cam_id = img.get("cam_id", 2)
            yaw = simulate_yaw_from_cam(cam_id)
            pitch = simulate_pitch()
            gaze_class = yaw_to_class(yaw)

            ann = {
                "id": ann_id,
                "image_id": img["id"],
                "cam_id": cam_id,
                "day_id": img["day_id"],
                "file_name": img["file_name"],
                "yaw": yaw,
                "pitch": pitch,
                "category_id": gaze_class,
                "bbox": [0, 0, img["width"], img["height"]]
            }
            new_annotations.append(ann)
            ann_id += 1

    final_data = {
        "annotations": new_annotations,
        "info": {
            "generated": True,
            "source": "synthetic-label-generator",
            "total": len(new_annotations)
        }
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2)

    print(f"✅ Created synthetic labels for {len(new_annotations)} frames.")
    print(f"💾 Saved new annotation file to: {out_path}")

if __name__ == "__main__":
    main()
