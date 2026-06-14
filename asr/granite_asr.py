import logging
from functools import cached_property
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

LOCAL_MODEL_DIR = Path("ibm-granite/granite-speech-4.1-2b")
HF_MODEL_ID = "ibm-granite/granite-speech-4.1-2b"


def get_model_path() -> str:
    return str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else HF_MODEL_ID


class GraniteSpeechASR:
    def __init__(self, device: str = None, dtype: torch.dtype = torch.bfloat16):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._prompt = None

    @cached_property
    def model(self):
        if self._model is not None:
            return self._model
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        model_path = get_model_path()
        logger.info(f"Loading Granite Speech model from: {model_path}")
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._tokenizer = self._processor.tokenizer
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, device_map=self._device, dtype=self._dtype
        )
        chat = [{"role": "user", "content": "<|audio|>transcribe the speech with proper punctuation and capitalization."}]
        self._prompt = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self.model
        return self._processor

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.model
        return self._tokenizer

    @property
    def prompt(self):
        if self._prompt is None:
            self.model
        return self._prompt

    def transcribe_segment(self, audio: torch.Tensor, max_new_tokens: int = 360) -> str:
        inputs = self.processor(
            self.prompt, audio, device=self._device, return_tensors="pt"
        ).to(self._device)

        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
        )

        num_input_tokens = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0, num_input_tokens:].unsqueeze(0)
        text = self.tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )[0].strip()
        return text

    def release(self):
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
