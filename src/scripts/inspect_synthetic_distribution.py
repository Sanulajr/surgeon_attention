import json
import os
from collections import Counter
import matplotlib.pyplot as plt

SYNTH_PATH = "MVOR/annotations/camma_mvor_2018_synthetic.json"

CLASS_NAMES = ["Left", "Front-Left", "Front", "Front-Right", "Right"]

def main():
    if not os.path.exists(SYNTH_PATH):
        print(f"❌ Synthetic file not found: {SYNTH_PATH}")
        return

    with open(SYNTH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    anns = data.get("annotations", [])
    if not anns:
        print("❌ No annotations found in synthetic JSON.")
        return

    class_counts = Counter([ann["category_id"] for ann in anns])
    total = sum(class_counts.values())

    print(f"\n📊 Synthetic Label Distribution Summary")
    print(f"-------------------------------------")
    print(f"Total samples: {total}\n")

    for i, cls_name in enumerate(CLASS_NAMES):
        count = class_counts.get(i, 0)
        percent = (count / total * 100) if total else 0
        print(f"{cls_name:<12}: {count:5d} samples  ({percent:5.2f}%)")

    # Optional: visualize
    plt.bar(CLASS_NAMES, [class_counts.get(i, 0) for i in range(len(CLASS_NAMES))])
    plt.title("Synthetic Gaze Class Distribution")
    plt.xlabel("Gaze Class")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    plt.show()

    print("\n✅ Distribution inspection complete.")

if __name__ == "__main__":
    main()
