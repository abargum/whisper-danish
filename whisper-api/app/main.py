import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from model_manager import ModelManager
from transcriber import transcribe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

FINETUNED_MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH", "abargum/whisper-tiny-base")
FINETUNED_LANGUAGE   = os.getenv("FINETUNED_LANGUAGE", "da")

manager = ModelManager(
    finetuned_model_path=FINETUNED_MODEL_PATH,
    finetuned_language=FINETUNED_LANGUAGE,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up — no model pre-loaded (lazy loading).")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Whisper Transcription API",
    description="Transcribe audio using whisper-small or a fine-tuned Whisper model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "current_model": manager.current_model_type,
        "finetuned_language": manager.finetuned_lang,
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file (wav, mp3, flac, ogg, ...)"),
    language: str | None = Form(
        default=None,
        description="ISO-639-1 language code e.g. 'en', 'da'. Omit to auto-detect.",
    ),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Normalize empty string to None so auto-detection triggers correctly
    if not language or not language.strip():
        language = None

    t0 = time.perf_counter()
    
    try:
        result = transcribe(audio_bytes, manager, language=language)
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc))

    result["processing_time_seconds"] = round(time.perf_counter() - t0, 3)
    return JSONResponse(content=result)