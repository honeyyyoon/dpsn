from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import models, jobs, images
from backend.db import init_db
from backend.services.cleanup import cleanup_old_data
from backend.services.image_store import ensure_target_image

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_target_image()
    cleanup_old_data()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router, tags=["models"])
app.include_router(jobs.router, tags=["jobs"])
app.include_router(images.router, tags=["images"])
