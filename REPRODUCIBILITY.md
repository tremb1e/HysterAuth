# Reproducibility Notes

This document summarizes how to reproduce the pipeline and experiments described in the HysterAuth paper using the code in this repository.

## Datasets
- HMOG dataset (IEEE TIFS 2015). Not included in this repository. Please follow the dataset license and terms.
- Real-world enterprise dataset referenced in the paper is not included.

## Data layout
- Raw device uploads: `server/data_storage/raw_data/<device_id_hash>/*.jsonl`
- Processed windows: `server/data_storage/processed_data/window/<device_id_hash>/...`
- Model outputs and policies: `server/data_storage/models/<device_id_hash>/`
- HMOG source path: configure `HMOG_DATA_PATH` (or edit `hmog_data_path` in `server/src/config.py`).

## End-to-end steps
1) Preprocess raw uploads into synchronized windows:
```
cd server
python -m src.processing.cli --user <device_id_hash>
```

2) Train per-user VQGAN models and generate policy files:
```
cd server
python -m src.training.cli --user <device_id_hash>
```

3) Train token Transformer models (HMOG scripts):
```
cd server/ca_train
python hmog_vqgan_token_transformer_experiment.py --dataset-path <processed_window_root> --users <user_id>
```

4) Run offline inference with trained checkpoints:
```
cd server/ca_train
python hmog_token_auth_inference.py --csv-path <window_csv> --window-size <sec> --vqgan-checkpoint <path> --lm-checkpoint <path>
```

5) Adjust decision policies via `server/ca_config.toml` and per-user `best_lock_policy.json`.

## Notes
- PyTorch is required for training and inference; install it separately for your platform (CPU or CUDA).
- The server uses gRPC for streaming; ensure the Android client and server ports match.
