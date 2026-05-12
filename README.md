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

- Upload WSI files (`.svs`, `.ndpi`, etc.) and process them immediately
- Select multiple models simultaneously for parallel comparison
- Before / After thumbnail visualization
- Quantitative evaluation via SSIM · PSNR · FID metrics
- Track processing status per job (`GET /jobs/{job_id}`)

---

## Supported Models

| ID | Model | Type |
|----|-------|------|
| 1 | Reinhard | Classical |
| 2 | Macenko | Classical |
| 3 | Vahadane | Classical |
| 4 | StainGAN | Learning-based |
| 5 | StainNet | Learning-based |
| 6 | StainSWIN | Learning-based |

The model registry is managed via `config/models.json`.

---

## Project Structure

```
dpsn/
  ai/          Normalization pipelines and metrics
  backend/     FastAPI server
  frontend/    React + TypeScript UI (Vite)
  config/      Shared configuration (models.json)
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.13+ |
| Node.js | 24+ |
| Yarn | latest |
| CUDA | 11.8+ (for GPU inference) |
| OpenSlide | 4.0+ |

> Install OpenSlide: `pip install openslide-python openslide-bin`  
> Server environment: Ubuntu 22.04 · Intel Xeon Silver 4210 · RTX 2080 Ti × 4 · 128GB RAM

---

## AI

```
ai/
  pipelines/   Per-model inference pipelines (Reinhard, StainNet implemented)
  models/      Model definitions and training scripts (StainNet, StainGAN)
  metrics/     Image quality metrics (SSIM, PSNR, FID)
  wsi/         WSI loading, patching, and writing
  samplers/    Patch sampling strategies (GridPatchSampler + OD-based tissue masking)
  runtime/     Worker / Task abstraction
```

`Worker` receives a `Task` (source image, target image, model ID), dispatches it to the appropriate pipeline, and returns the normalized image along with quality metrics.

---

## Backend

### Installation

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running

```bash
source backend/venv/bin/activate
PYTHONPATH=. uvicorn backend.main:app --reload
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/models` | List all registered models |
| `GET` | `/models/{model_id}` | Get details of a specific model |
| `POST` | `/jobs` | Create a normalization job (WSI upload + model selection) |
| `GET` | `/jobs/{job_id}` | Poll job status |
| `GET` | `/jobs/{job_id}/results` | Retrieve results and metrics |

> `POST /jobs` request format: `multipart/form-data` — `image` file + `model_ids` (comma-separated string)  
> When multiple models are selected, a separate job is created per model.

---

## Frontend

### Installation & Dev Server

```bash
cd frontend
yarn
yarn dev   # http://localhost:5173
```

### Build

```bash
yarn build
```

---

## Team

| Name | Role |
|------|------|
| Shiheon Yoon | Model serving API (FastAPI), AI runtime integration, frontend (UI, job submission, results visualization) |
| Yebin Pyun | AI model training & optimization, inference pipeline implementation |
| Jiseong Lee | WSI & metrics modules, classical algorithm optimization, classification-based evaluation metrics |

---

## References

Hoque, M. Z., Keskinarkaus, A., Nyberg, P., & Seppänen, T. (2024).  
*Stain normalization methods for histopathology image analysis: A comprehensive review.*  
Information Fusion. https://doi.org/10.1016/j.inffus.2024.102198