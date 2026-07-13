"""Load and run an exported English-German Transformer Hub bundle."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_model
from tokenizers import Tokenizer
import torch

from modeling_transformer import Transformer


def _pad(sequences: list[list[int]], pad_token_id: int) -> torch.Tensor:
    max_length = max(map(len, sequences))
    return torch.tensor(
        [
            sequence + [pad_token_id] * (max_length - len(sequence))
            for sequence in sequences
        ],
        dtype=torch.long,
    )


class TransformerTranslator:
    """Small inference wrapper around the educational Transformer."""

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        config: dict[str, object],
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.bos_token_id = tokenizer.token_to_id("[BOS]")
        self.eos_token_id = tokenizer.token_to_id("[EOS]")
        if None in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
            raise ValueError("Tokenizer must define [PAD], [BOS], and [EOS]")

    @classmethod
    def from_pretrained(
        cls,
        model_directory: str | Path,
        device: str | torch.device = "auto",
    ) -> TransformerTranslator:
        model_directory = Path(model_directory)
        config = json.loads((model_directory / "config.json").read_text())
        model_config = config["model_config"]
        tokenizer = Tokenizer.from_file(str(model_directory / "tokenizer.json"))
        if device == "auto":
            resolved_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            resolved_device = torch.device(device)
        model = Transformer(**model_config)
        missing, unexpected = load_model(
            model,
            model_directory / "model.safetensors",
            device="cpu",
            strict=True,
        )
        if missing or unexpected:
            raise ValueError(
                f"Weight mismatch: missing={missing}, unexpected={unexpected}"
            )
        return cls(model, tokenizer, config, resolved_device)

    def _encode(self, text: str) -> list[int]:
        context_length = int(self.config["model_config"]["context_length"])
        token_ids = self.tokenizer.encode(text).ids
        if len(token_ids) > context_length:
            token_ids = token_ids[:context_length]
            token_ids[-1] = self.eos_token_id
        return token_ids

    @torch.inference_mode()
    def translate(
        self,
        texts: str | list[str],
        max_new_tokens: int | None = None,
    ) -> str | list[str]:
        single_text = isinstance(texts, str)
        text_batch = [texts] if single_text else texts
        if not text_batch:
            return []
        context_length = int(self.config["model_config"]["context_length"])
        max_new_tokens = max_new_tokens or context_length
        if not 2 <= max_new_tokens <= context_length:
            raise ValueError(
                f"max_new_tokens must be between 2 and {context_length}"
            )

        source_ids = _pad(
            [self._encode(text) for text in text_batch], self.pad_token_id
        ).to(self.device)
        generated = torch.full(
            (len(text_batch), 1),
            self.bos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        finished = torch.zeros(
            len(text_batch), dtype=torch.bool, device=self.device
        )
        source_padding_mask = source_ids.eq(self.pad_token_id)
        for _ in range(max_new_tokens - 1):
            outputs = self.model(
                source_ids=source_ids,
                decoder_input_ids=generated,
                source_padding_mask=source_padding_mask,
                decoder_padding_mask=generated.eq(self.pad_token_id),
            )
            next_tokens = outputs["logits"][:, -1].argmax(dim=-1)
            next_tokens = torch.where(finished, self.pad_token_id, next_tokens)
            generated = torch.cat((generated, next_tokens[:, None]), dim=1)
            finished |= next_tokens.eq(self.eos_token_id)
            if finished.all():
                break

        translations = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated.cpu().tolist()
        ]
        return translations[0] if single_text else translations
