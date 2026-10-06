# Acacia Model Fine-Tuning — Deployment

Dockerized deployment of the Node/Express web app + Detectron2 (PyTorch/CUDA) training backend.

## What's in the image
- **Node 20** runs the Express server (`index.js`) on port `5000`.
- **PyTorch 2.5.0 + CUDA 12.1 devel** base image (Python 3.11, CUDA toolkit and gcc), the
  same versions as the local `detectron_env`. Detectron2 needs the CUDA compiler (nvcc)
  matching PyTorch's CUDA version to build its ops, which is why the `-devel` image is used.
- **Detectron2** installed from GitHub with
  `pip install "git+https://github.com/facebookresearch/detectron2.git@<commit>"`,
  pinned to commit `02b5c4e` (the same commit as the local
  `D:\Project\ModelTraining\detectron2` checkout).
- The **ImageNet R-50 backbone** (`R-50.pkl`, ~100 MB) used for transfer learning, so the
  server can train without internet access.
- Python deps from `training/requirements.txt` (torch/torchvision come from the base image).

## Prerequisites (host)
- Docker + Docker Compose
- NVIDIA GPU with recent drivers (CUDA 12.1 support)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (so the container can access the GPU)

Verify GPU access works:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Build & run
Copy the project to the server (e.g. `git clone`), then from the project folder:
```bash
docker compose up --build -d
```
The app is then available at **http://<server>:5000**.

The first build takes a while (roughly 15–30 minutes): it downloads the PyTorch images
and compiles Detectron2's CUDA ops. The internet is only needed during the build.

View logs (training output streams here):
```bash
docker compose logs -f
```

Stop:
```bash
docker compose down
```

Update after pulling new code:
```bash
git pull
docker compose up --build -d
```

## Configuration
| Var          | Default  | Purpose                                         |
|--------------|----------|-------------------------------------------------|
| `PORT`       | `5000`   | Server port                                     |
| `PYTHON_BIN` | `python` | Python interpreter used to spawn training       |
| `MPLBACKEND` | `Agg`    | Headless matplotlib backend (no display needed) |

Build arguments (`docker compose build --build-arg NAME=value`):
| Arg                     | Default                  | Purpose                                         |
|-------------------------|--------------------------|-------------------------------------------------|
| `TORCH_CUDA_ARCH_LIST`  | `7.0;7.5;8.0;8.6;8.9;9.0` | GPU architectures Detectron2 is compiled for. Set it to just the server's GPU (e.g. `8.6`) for a faster build. |
| `DETECTRON2_REF`        | `02b5c4e…`               | Detectron2 commit to build                      |

To run on another port, change both sides of `ports:` and `PORT` in `docker-compose.yml`.

## Data persistence
These host folders are mounted into the container, so they survive restarts and rebuilds:
| Host               | Container             | Contents                                      |
|--------------------|-----------------------|-----------------------------------------------|
| `./public/uploads` | `/app/public/uploads` | Uploaded datasets (images + annotation JSON)  |
| `./outputs`        | `/app/outputs`        | Training runs: weights, metrics, logs, plots  |

To move existing datasets and runs to the server, copy these two folders into the project
folder before starting the container.

## Running locally without Docker
Point the app at the conda environment that has Detectron2:
```powershell
$env:PYTHON_BIN = "C:\Users\MS1\anaconda3\envs\detectron_env\python.exe"
node index.js
```

## Notes
- `shm_size: 8gb` is needed because PyTorch DataLoader workers share batches through
  `/dev/shm`; Docker's 64 MB default makes training crash with a "bus error".
- `init: true` runs a small init process as PID 1, so Python training processes that are
  stopped or finish are cleaned up properly.
- Training runs inside the web container, so `docker compose down` (or a rebuild) stops
  any training that is in progress.
- The container runs as root, so files it writes into `public/uploads` and `outputs` on
  the host are owned by root.
