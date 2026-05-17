import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.config import settings
from src.storage.repository import AnalysisRepository
from src.core.analyzer import run_analysis

logger = logging.getLogger(__name__)

repo = AnalysisRepository()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await repo.init_db()
    logger.info("api.startup")
    yield
    logger.info("api.shutdown")


app = FastAPI(title="AI Food Analyzer", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type '{image.content_type}'. Only JPEG and PNG accepted.",
        )

    data = await image.read()

    if len(data) > settings.max_image_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data)} bytes). Maximum is {settings.max_image_bytes} bytes.",
        )

    if len(data) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    logger.info("api.analyze.received", extra={"filename": image.filename, "size": len(data)})

    suffix = ".jpg" if image.content_type == "image/jpeg" else ".png"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        record = await run_analysis(tmp_path)
        saved = await repo.save(record)

        logger.info("api.analyze.done", extra={"id": saved.id})
        return JSONResponse(content=saved.model_dump(mode="json"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("api.analyze.failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/history")
async def history(limit: int = 20) -> list:
    records = await repo.list_all(limit=limit)
    return [r.model_dump(mode="json") for r in records]