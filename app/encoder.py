"""The model. The only module that imports torch, and only when it loads."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import NATIVE_DIMENSIONS, PROMPTS, Settings


class Encoder(Protocol):
    """What the API needs: unit vectors of NATIVE_DIMENSIONS, and tokens used."""

    def encode(self, texts: list[str], input_type: str) -> tuple[np.ndarray, int]:
        """Embed `texts` as "query" or "document": (n, 2048) float32 rows of
        length 1, in the order given, and the tokens processed."""
        ...


class VoyageEncoder:
    """voyage-4-nano as its model card loads it: the Qwen3 bidirectional
    model (the revision's own code, trust_remote_code), the model's prompts
    in front of the text, mean pooling over every token, then L2."""

    def __init__(self, settings: Settings) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(settings.threads)
        self._torch = torch
        self._settings = settings
        self._tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
        dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float32
        self._model = AutoModel.from_pretrained(
            settings.model_path, trust_remote_code=True, dtype=dtype
        ).eval()

    def encode(self, texts: list[str], input_type: str) -> tuple[np.ndarray, int]:
        # Grad mode is per thread, and requests run on the server's threads.
        with self._torch.inference_mode():
            return self._encode(texts, input_type)

    def _encode(self, texts: list[str], input_type: str) -> tuple[np.ndarray, int]:
        torch = self._torch
        prompt = PROMPTS[input_type]
        settings = self._settings
        encoded = self._tokenizer(
            [prompt + text for text in texts],
            truncation=True,
            max_length=settings.max_tokens,
            add_special_tokens=True,
        )["input_ids"]
        # Longest last, so a batch's padding is small; a batch is at most
        # batch_size texts and batch_tokens padded tokens.
        order = sorted(range(len(texts)), key=lambda i: len(encoded[i]))
        out = np.zeros((len(texts), NATIVE_DIMENSIONS), dtype=np.float32)
        position = 0
        while position < len(order):
            end = position + 1
            while (
                end < len(order)
                and end - position < settings.batch_size
                and (end - position + 1) * len(encoded[order[end]]) <= settings.batch_tokens
            ):
                end += 1
            rows = order[position:end]
            width = max(len(encoded[i]) for i in rows)
            pad = self._tokenizer.pad_token_id or 0
            ids = torch.full((len(rows), width), pad, dtype=torch.long)
            mask = torch.zeros((len(rows), width), dtype=torch.long)
            for r, i in enumerate(rows):
                ids[r, : len(encoded[i])] = torch.tensor(encoded[i])
                mask[r, : len(encoded[i])] = 1
            hidden = self._model(input_ids=ids, attention_mask=mask).last_hidden_state
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1).float() / weights.sum(1).float().clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out[rows] = pooled.numpy()
            position = end
        return out, sum(len(ids) for ids in encoded)
