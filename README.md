# DPSN — Digital Pathology Stain Normalizer

> In collaboration with [INFINITT Healthcare](https://www.infinitt.com)  
> Seoul National University · Creative Integrated Design 2 (2026-1)

A benchmarking web platform for comparing multiple stain normalization models on whole slide images (WSI) side by side.

---

## Background

Digital pathology slides vary in color appearance depending on staining reagents, scanner equipment, and institution — causing AI diagnostic models to perform inconsistently across hospitals.  
**Stain Normalization** is a preprocessing technique that reduces this domain gap. DPSN provides a unified platform to objectively compare and evaluate multiple normalization methods.

---

## Features

- Upload WSI files (`.svs`, `.ndpi`, `.tiff`, etc.) and process them immediately
- Select multiple models simultaneously for parallel comparison
- Before / After image visualization with side-by-side dashboard
- Quantitative evaluation via **5 metrics**: SSIM · PSNR · FID · Gaussian Color Distance · Gaussian Color Gain
- Real-time job progress tracking with GPU device queue
- Job history persisted in SQLite (survives server restarts)
- Bulk download of normalized results as ZIP

---

## Supported Models

| ID | Model | Type | Description |
|----|-------|------|-------------|
| 1 | Reinhard | Classical | Color distribution transfer via statistical matching |
| 2 | Macenko | Classical | Stain vector estimation and decomposition |
| 3 | Vahadane | Classical | Structure-preserving stain separation |
| 4 | StainGAN | Learning-based | CycleGAN-based Image-to-Image translation |
| 5 | StainNet | Learning-based | Lightweight CNN-based normalization |
| 6 | StainSWIN | Learning-based | Swin Transformer-based model |
| 7 | MultiStainCycleGAN | Learning-based | Many-to-one CycleGAN across multiple stain domains (custom) |

The model registry is managed via `config/models.json`.

---

## Project Structure

```
dpsn/
├── ai/                    AI normalization module
│   ├── pipelines/         Per-model inference pipelines (base.py interface)
│   ├── models/            Model definitions and training scripts
│   ├── metrics/           Image quality metrics (SSIM, PSNR, FID, Gaussian Color Distance)
│   ├── wsi/               WSI loading, patching, and writing (OpenSlide / tifffile)
│   ├── samplers/          Patch sampling strategies (grid sampler, tissue masking)
│   └── runtime/           Worker / Task abstraction
├── backend/               FastAPI REST API server
│   ├── routers/           API endpoints (models, jobs, images)
│   ├── services/          Job runner (ThreadPoolExecutor + GPU queue), image store
│   └── db.py              SQLite database (WAL mode)
├── frontend/              React + TypeScript SPA (Vite)
│   └── src/components/    UI components (Sidebar, ConfigPanel, ResultsViews, etc.)
└── config/                Shared configuration (models.json)
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| Node.js | 24+ |
| CUDA | 11.8+ (for GPU inference, optional) |
| OpenSlide | 4.0+ |

> **GPU is recommended** for deep learning models but not required.  
> Classical algorithms (Reinhard, Macenko, Vahadane) run on CPU without issues.  
> For CPU-only environments, use small input images for reasonable processing time.

---

## Getting Started

### Backend

```bash
# Create virtual environment and install dependencies
python -m venv backend/venv
source backend/venv/bin/activate
pip install -r requirements.txt          # local (includes torch)
# or
pip install -r requirements-server.txt  # server (torch installed separately)

# Run the server
PYTHONPATH=. uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Data directory defaults to `/mnt/Disk1/dpsn_data`. Override with:

```bash
export DPSN_DATA_DIR=/path/to/your/data
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

By default the frontend connects to `http://10.10.40.182:8000`. Override with:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

### Model Checkpoints

Deep learning models require pre-trained checkpoints placed in `ai/checkpoints/`:

```
ai/checkpoints/
├── staingan/     staingan_aperio_to_hamamatsu_latest.pth
├── stainnet/     stainnet_aperio_to_hamamatsu_latest.pth
└── stainswin/    stainswin_aperio_to_hamamatsu_latest.pth
```

> Checkpoints are not included in this repository due to file size.  
> Classical algorithms (Reinhard, Macenko, Vahadane) work without checkpoints.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/models` | List all registered models |
| `GET` | `/models/{model_id}` | Get details of a specific model |
| `POST` | `/jobs` | Create normalization jobs (WSI upload + model selection) |
| `POST` | `/jobs/add` | Add models to an existing job group |
| `GET` | `/jobs` | List all job groups |
| `GET` | `/jobs/{job_id}` | Poll job status (`pending` / `running` / `done` / `failed`) |
| `GET` | `/jobs/{job_id}/results` | Retrieve result image ID and metrics |
| `DELETE` | `/jobs/{job_id}` | Cancel or delete a job |
| `GET` | `/images/{image_id}` | Serve stored image file |
| `GET` | `/images/target` | Serve target reference image |

> `POST /jobs`: `multipart/form-data` — `image` file + `model_ids` (comma-separated).  
> When multiple models are selected, a separate job is created per model.

---

## Architecture

```
Frontend (React)  ⟷  FastAPI  ⟷  Job Runner  ⟷  Worker  ⟷  Pipeline
                                     │                         ├── Patch Sampler
                                   SQLite                      └── Metric
```

- **Job Runner**: Manages job lifecycle with `ThreadPoolExecutor`. GPU device queue distributes jobs across available GPUs (cuda:1, cuda:2, cuda:3).
- **Worker**: Receives a `Task`, dispatches it to the appropriate pipeline via `PIPELINE_MAP`, returns `TaskResult` with normalized image and metrics.
- **Pipeline**: Each model implements `base.py` interface. Calls Patch Sampler for preprocessing and Metric for evaluation during processing.

---

## Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| SSIM | ≥ 0.90 | Structural Similarity Index Measure |
| PSNR | ≥ 22 dB | Peak Signal-to-Noise Ratio |
| FID | ≤ 20 | Fréchet Inception Distance |
| Gaussian Color Distance | ≤ 0.50 | GMM-based color distribution distance (custom) |
| Gaussian Color Gain | > 0 | Relative improvement in color distance after normalization (custom) |

Gaussian Color Distance and Gaussian Color Gain are custom metrics developed for this project, using HSV color space with GMM (k=3) to measure color distribution similarity between images.

---

## Team

| Name | Role |
|------|------|
| Shiheon Yoon | Model Serving API, AI Runtime, Frontend, system integration |
| Yebin Pyun | AI model training, inference pipelines, Patch Sampler |
| Jiseong Lee | WSI & metrics modules, classical algorithms, Gaussian Color Distance metric |

---

## References

- Hoque, M. Z., Keskinarkaus, A., Nyberg, P., & Seppänen, T. (2024). *Stain normalization methods for histopathology image analysis: A comprehensive review.* Information Fusion. https://doi.org/10.1016/j.inffus.2024.102198
