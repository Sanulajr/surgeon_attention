import json
import os

def yaw_pitch_to_gaze_class(yaw, pitch):
    """Convert yaw (left-right) and pitch (up-down) into a discrete gaze class."""
    if yaw < -30:
        return 0  # Left
    elif yaw < -10:
        return 1  # Front-Left
    elif yaw < 10:
        return 2  # Front
    elif yaw < 30:
        return 3  # Front-Right
    else:
        return 4  # Right

def main():
    src_path = "MVOR/annotations/camma_mvor_2018.json"
    out_path = "MVOR/annotations/camma_mvor_2018_labeled.json"

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    added = 0
    for ann in data.get("annotations", []):
        yaw = ann.get("yaw", 0)
        pitch = ann.get("pitch", 0)
        label = yaw_pitch_to_gaze_class(yaw, pitch)
        ann["category_id"] = label
        added += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Added gaze labels to {added} annotations.")
    print(f"💾 Saved updated file: {out_path}")

if __name__ == "__main__":
    main()
