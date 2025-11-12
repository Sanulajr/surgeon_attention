# MVOR Attention Project (PyTorch)

Train a model to predict head pose (yaw, pitch) and gaze class from the multi-view MVOR dataset.

## Quickstart

```
# From workspace root (where MVOR/ exists)
python -m src.scripts.train_attention_model --config src/config.yaml
python -m src.scripts.evaluate_attention_model --config src/config.yaml --model_path outputs/best_attention_model.pth --visualize
```

- Put the dataset at `MVOR/` with `dataset/` and `annotations/camma_mvor_2018.json` present.
- The loader tries multiple subfolders for images (e.g., `dataset`, `dataset/color`, etc.).

## Config

Edit `src/config.yaml` to change hyperparameters, augmentation, and model options.

## Notes

- If explicit head pose or gaze labels are missing, the loader **heuristically infers** them from 2D keypoints.
- Multi-view fusion uses attention pooling by default.
