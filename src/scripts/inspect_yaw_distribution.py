import json
import numpy as np

path = "MVOR/annotations/camma_mvor_2018.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

yaws = [ann["yaw"] for ann in data["annotations"] if "yaw" in ann]
pitches = [ann["pitch"] for ann in data["annotations"] if "pitch" in ann]

print(f"Total samples: {len(yaws)}")
print(f"Yaw range: {min(yaws):.2f} to {max(yaws):.2f}")
print(f"Pitch range: {min(pitches):.2f} to {max(pitches):.2f}")

# Optional: show histogram
bins = np.linspace(-60, 60, 13)
hist, _ = np.histogram(yaws, bins=bins)
print("Yaw histogram bins:", bins)
print("Counts:", hist.tolist())
