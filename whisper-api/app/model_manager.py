import time
import logging
from enum import Enum
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    BASE = "tiny"          
    FINETUNED = "finetuned"


class ModelManager:
    def __init__(self, finetuned_model_path: str, finetuned_language: str):
        self.finetuned_model_path = finetuned_model_path
        self.finetuned_language = finetuned_language.lower()

        self._current_type: Optional[ModelType] = None
        self._processor: Any = None
        self._model: Any = None

    def get_model(self, model_type: ModelType):
        """Return (processor, model), loading/swapping if necessary."""
        if self._current_type == model_type:
            logger.info(f"[ModelManager] '{model_type}' already loaded — reusing.")
            return self._processor, self._model

        if self._current_type is not None:
            logger.info(f"[ModelManager] Unloading '{self._current_type}'...")
            self._unload()

        t0 = time.perf_counter()
        logger.info(f"[ModelManager] Loading '{model_type}'...")

        if model_type == ModelType.FINETUNED:
            self._load_finetuned()
        else:
            self._load_base()

        elapsed = time.perf_counter() - t0
        self._current_type = model_type
        logger.info(f"[ModelManager] '{model_type}' ready — load time: {elapsed:.2f}s")
        return self._processor, self._model

    @property
    def finetuned_lang(self) -> str:
        return self.finetuned_language

    @property
    def current_model_type(self) -> Optional[ModelType]:
        return self._current_type

    def _load_finetuned(self):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self._processor = WhisperProcessor.from_pretrained(self.finetuned_model_path)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.finetuned_model_path
        )
        self._model.eval()

    def _load_base(self):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self._processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self._model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny"
        )
        self._model.eval()

    def _unload(self):
        import torch, gc
        self._processor = None
        self._model = None
        gc.collect()
        if torch.cuda.is_available():   
            torch.cuda.empty_cache()
        self._current_type = None