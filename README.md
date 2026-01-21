# HysterAuth

HysterAuth is a robust, low-friction continuous authentication framework for mobile devices. This repository is the companion implementation for the paper "HysterAuth: A Robust and Frictionless Continuous Authentication Framework via Diffusion Augmentation and Hysteresis Decision".

## Paper summary
HysterAuth reframes continuous authentication as a token-evidence sequential decision problem. It combines:
- Modality-aware diffusion augmentation for few-shot enrollment
- Vector quantization to discrete behavioral tokens
- Causal Transformer sequence modeling for likelihood-based scoring
- Hysteresis-based evidence accumulation to suppress decision boundary oscillations

Reported results include FRR 0.011 percent, about 2 false alarms per hour, and sub-second intrusion detection latency on HMOG and real-world enterprise deployments.

## System overview (paper-aligned)
On-device (Android):
- Collect accelerometer , gyroscope  and magnetometer at 100 Hz
- Batch, compress (LZ4 or GZIP), and envelope-encrypt payloads
- Stream packets over gRPC with HMAC-hashed device/user/app identifiers

Server (Python):
- Decrypt and decompress batches, validate metadata, and store raw JSONL
- Preprocess to resampled, synchronized windows and z-score normalize
- Tokenize windows via Vector Quantization, score with a causal Transformer LM
- Accumulate log-odds evidence with HysterAuth hysteresis to emit interruptions only on state transitions

## Repository layout
- ContinuousAuthentication/ - Android client for sensor collection and encrypted gRPC streaming
- server/ - ingestion, preprocessing, training, inference, and policy management
  - server/src/paper_modules - reference implementations of diffusion, VQ, Transformer, HysterAuth
  - server/ca_train - training and evaluation scripts for HMOG and token models
  - server/scripts/demo_paper_and_server.py - runnable demo of paper modules and server pipeline

## Quick start

### Server
```
cd server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ml.txt
# Install PyTorch separately for your platform (CPU or CUDA)
python -m src.main
```

Default gRPC listens on 0.0.0.0:10500. The HTTP health endpoint (`/health`) is available when HTTP is enabled and uses a different port from gRPC.

### Demo (paper modules + pipeline)
```
cd server
python scripts/demo_paper_and_server.py
```

### Android client
- Open `ContinuousAuthentication` in Android Studio.
- Configure server host and port in the in-app Server Config screen.
- Build and run on Android 11+ (minSdk 30).
- Provide privacy consent before starting background collection.

## Data processing and training
Raw uploads are stored under `server/data_storage/raw_data/<device_id_hash>/` as JSONL batches.

1) Preprocess raw data into windows:
```
cd server
python -m src.processing.cli --user <device_id_hash>
```

2) Train per-user models (VQ and policy selection):
```
cd server
python -m src.training.cli --user <device_id_hash>
```

Training artifacts and policies are saved under `server/data_storage/models/<device_id_hash>/`.

## Configuration
- `server/ca_config.toml` controls window sizes, overlap, and decision targets (vote or K-reject rules).
- `.env` or environment variables override `server/src/config.py` settings (ports, paths, TLS, logging).

## Security and privacy
- Only zero-permission inertial sensors are collected (acc, gyr, mag).
- Device/user/app identifiers are HMAC hashed before transmission.
- Payloads are compressed and envelope-encrypted on device.
- Server deployment is designed for private infrastructure to keep data local.

## Testing
```
cd server
pytest
```
