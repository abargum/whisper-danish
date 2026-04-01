import io
import logging
import numpy as np
import torch
import librosa

from model_manager import ModelManager, ModelType

logger = logging.getLogger(__name__)

TARGET_SR = 16_000 


def _load_audio(audio_bytes: bytes) -> np.ndarray:
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SR, mono=True)
    return audio


def detect_language(audio: np.ndarray, manager: ModelManager) -> str:
    processor, model = manager.get_model(ModelType.BASE)
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    input_features = inputs.input_features

    with torch.no_grad():
        predicted_ids = model.detect_language(input_features)

    top_token_id = predicted_ids[0].item()
    lang = processor.tokenizer.decode(top_token_id).strip("<|>")

    logger.info(f"[Transcriber] Detected language: '{lang}'")
    return lang


def transcribe(
    audio_bytes: bytes,
    manager: ModelManager,
    language: str | None = None,
) -> dict:
    
    audio = _load_audio(audio_bytes)

    if language is None:
        detected_lang = detect_language(audio, manager)
        language = detected_lang if detected_lang else "en"  # fallback
    else:
        language = language.lower()

    if language == manager.finetuned_lang:
        model_type = ModelType.FINETUNED
        logger.info(f"[Transcriber] Routing to FINETUNED model (language=\'{language}\')")
    else:
        model_type = ModelType.BASE
        logger.info(f"[Transcriber] Routing to BASE model (language=\'{language}\')")

    processor, model = manager.get_model(model_type)

    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    input_features = inputs.input_features

    forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_ids,
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return {
        "text": transcription.strip(),
        "language": language,
        "model_used": model_type.value,
    }