"""Train the WMT14 English-German Transformer outside Jupyter.

Run this script in tmux so training survives SSH and VS Code disconnects. It
automatically resumes from checkpoints/transformer_en_de-v2/latest.pt.
"""

from __future__ import annotations

import argparse
from array import array
from hashlib import sha256
from itertools import islice, zip_longest
import json
import math
from pathlib import Path
import random
import signal
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from tqdm.auto import tqdm


EMBEDDING_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
CONTEXT_LENGTH = 256
DROPOUT_RATE = 0.1
LABEL_SMOOTHING = 0.1


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("Could not find repo root containing pyproject.toml")


class ParallelTextDataset(Dataset):
    def __init__(
        self,
        source_path: Path,
        target_path: Path,
        tokenizer: Tokenizer,
        max_seq_len: int,
        max_examples: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id = tokenizer.token_to_id("[EOS]")

        pairs = []
        with (
            source_path.open(encoding="utf-8") as source_file,
            target_path.open(encoding="utf-8") as target_file,
        ):
            lines = zip_longest(source_file, target_file)
            if max_examples is not None:
                lines = islice(lines, max_examples)
            for source_line, target_line in lines:
                if source_line is None or target_line is None:
                    raise ValueError("Source and target files must have the same number of lines")
                source_text = source_line.rstrip()
                target_text = target_line.rstrip()
                if source_text and target_text:
                    pairs.append((source_text, target_text))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode(self, text: str) -> list[int]:
        token_ids = self.tokenizer.encode(text).ids
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len - 1] + [self.eos_token_id]
        return token_ids

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        source_text, target_text = self.pairs[index]
        return {
            "source_ids": self._encode(source_text),
            "target_ids": self._encode(target_text),
        }


class IndexedParallelTextDataset(Dataset):
    """Parallel text backed by byte offsets instead of millions of Python strings."""

    def __init__(
        self,
        source_path: Path,
        target_path: Path,
        tokenizer: Tokenizer,
        max_seq_len: int,
    ) -> None:
        self.source_path = source_path
        self.target_path = target_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id = tokenizer.token_to_id("[EOS]")
        self.source_offsets = self._line_offsets(source_path)
        self.target_offsets = self._line_offsets(target_path)
        if len(self.source_offsets) != len(self.target_offsets):
            raise ValueError("Source and target files have different line counts")
        self._source_file = None
        self._target_file = None

    @staticmethod
    def _line_offsets(path: Path) -> array:
        offsets = array("Q")
        with path.open("rb") as text_file:
            while True:
                offset = text_file.tell()
                line = text_file.readline()
                if not line:
                    break
                offsets.append(offset)
        return offsets

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_source_file"] = None
        state["_target_file"] = None
        return state

    def __len__(self) -> int:
        return len(self.source_offsets)

    def _open_files(self) -> None:
        if self._source_file is None:
            self._source_file = self.source_path.open("rb")
            self._target_file = self.target_path.open("rb")

    def _encode(self, text: str) -> list[int]:
        token_ids = self.tokenizer.encode(text).ids
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len - 1] + [self.eos_token_id]
        return token_ids

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        self._open_files()
        self._source_file.seek(self.source_offsets[index])
        self._target_file.seek(self.target_offsets[index])
        source = self._source_file.readline().decode("utf-8").rstrip()
        target = self._target_file.readline().decode("utf-8").rstrip()
        return {
            "source_ids": self._encode(source),
            "target_ids": self._encode(target),
        }

    def text_pair(self, index: int) -> tuple[str, str]:
        self._open_files()
        self._source_file.seek(self.source_offsets[index])
        self._target_file.seek(self.target_offsets[index])
        return (
            self._source_file.readline().decode("utf-8").rstrip(),
            self._target_file.readline().decode("utf-8").rstrip(),
        )


class TokenBucketBatchSampler(Sampler[list[int]]):
    """Deterministic shuffled batches bounded by padded source+target tokens."""

    def __init__(
        self,
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        max_tokens: int,
        seed: int,
        epoch: int,
        bucket_size: int = 2_048,
        max_sentences: int = 256,
    ) -> None:
        if len(source_lengths) != len(target_lengths):
            raise ValueError("Source and target length caches differ")
        self.batches = self._make_batches(
            source_lengths,
            target_lengths,
            max_tokens,
            seed,
            epoch,
            bucket_size,
            max_sentences,
        )

    @staticmethod
    def _make_batches(
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        max_tokens: int,
        seed: int,
        epoch: int,
        bucket_size: int,
        max_sentences: int,
    ) -> list[list[int]]:
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(source_lengths), generator=generator).tolist()
        batches: list[list[int]] = []
        for start in range(0, len(order), bucket_size):
            bucket = order[start : start + bucket_size]
            bucket.sort(
                key=lambda index: int(source_lengths[index] + target_lengths[index])
            )
            batch: list[int] = []
            max_source = max_target = 0
            for index in bucket:
                source_length = int(source_lengths[index])
                target_length = int(target_lengths[index])
                next_max_source = max(max_source, source_length)
                next_max_target = max(max_target, target_length)
                padded_tokens = (next_max_source + next_max_target) * (len(batch) + 1)
                if batch and (
                    padded_tokens > max_tokens or len(batch) >= max_sentences
                ):
                    batches.append(batch)
                    batch = []
                    max_source = max_target = 0
                batch.append(index)
                max_source = max(max_source, source_length)
                max_target = max(max_target, target_length)
            if batch:
                batches.append(batch)
        permutation = torch.randperm(len(batches), generator=generator).tolist()
        return [batches[index] for index in permutation]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def source_grouped_split(
    dataset: ParallelTextDataset,
    validation_examples: int,
    seed: int,
) -> tuple[Subset, Subset, list[int]]:
    """Split complete English-source groups into train and validation sets."""
    if validation_examples <= 0 or validation_examples >= len(dataset):
        raise ValueError("--validation-examples must be between 1 and dataset size - 1")

    source_groups: dict[str, list[int]] = {}
    for index, (source_text, _target_text) in enumerate(dataset.pairs):
        source_groups.setdefault(source_text, []).append(index)

    groups = list(source_groups.values())
    random.Random(seed).shuffle(groups)
    validation_indices: list[int] = []
    for group in groups:
        if len(validation_indices) + len(group) <= validation_examples:
            validation_indices.extend(group)
            if len(validation_indices) == validation_examples:
                break

    if not validation_indices:
        raise ValueError("Could not construct a source-grouped validation split")
    validation_indices.sort()
    validation_set = set(validation_indices)
    train_indices = [
        index for index in range(len(dataset)) if index not in validation_set
    ]
    return (
        Subset(dataset, train_indices),
        Subset(dataset, validation_indices),
        validation_indices,
    )


class TranslationCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def _pad(self, sequences: list[list[int]]) -> torch.Tensor:
        max_len = max(map(len, sequences))
        padded = [
            sequence + [self.pad_token_id] * (max_len - len(sequence))
            for sequence in sequences
        ]
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        source_ids = self._pad([example["source_ids"] for example in examples])
        target_ids = self._pad([example["target_ids"] for example in examples])
        decoder_input_ids = target_ids[:, :-1]
        return {
            "source_ids": source_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": target_ids[:, 1:],
            "source_padding_mask": source_ids.eq(self.pad_token_id),
            "decoder_padding_mask": decoder_input_ids.eq(self.pad_token_id),
        }


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, context_length: int, embedding_dim: int) -> None:
        super().__init__()
        position = torch.arange(context_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2)
            * (-torch.log(torch.tensor(10_000.0)) / embedding_dim)
        )
        pe = torch.zeros(context_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe[positions]


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        is_crossattention: bool = False,
        is_causal: bool = False,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if is_crossattention and is_causal:
            raise ValueError("Cross-attention cannot be causal")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.is_crossattention = is_crossattention
        self.is_causal = is_causal
        self.w_q = nn.Linear(embedding_dim, embedding_dim)
        self.w_k = nn.Linear(embedding_dim, embedding_dim)
        self.w_v = nn.Linear(embedding_dim, embedding_dim)
        self.w_o = nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = nn.Dropout(dropout_rate)
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1
            )
            self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, embedding_dim = hidden_states.shape
        if self.is_crossattention:
            if key_value_states is None:
                raise ValueError("Cross-attention requires key_value_states")
        else:
            if key_value_states is not None:
                raise ValueError("Self-attention does not accept key_value_states")
            key_value_states = hidden_states

        key_value_len = key_value_states.shape[1]
        q = self.w_q(hidden_states)
        k = self.w_k(key_value_states)
        v = self.w_v(key_value_states)
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_value_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_value_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if self.is_causal:
            causal_mask = self.causal_mask[:query_len, :key_value_len]
            attn_scores = attn_scores.masked_fill(
                causal_mask[None, None, :, :], float("-inf")
            )
        if padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                padding_mask[:, None, None, :], float("-inf")
            )
        attn_weights = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        output = attn_weights @ v
        output = output.transpose(1, 2).contiguous().view(
            batch_size, query_len, embedding_dim
        )
        return self.w_o(output)


class EncoderBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, num_heads: int, context_length: int, dropout_rate: float
    ) -> None:
        super().__init__()
        self.self_attention = MultiheadAttention(
            embedding_dim, num_heads, context_length, dropout_rate=dropout_rate
        )
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn_output = self.self_attention(hidden_states, padding_mask=padding_mask)
        hidden_states = self.layer_norm1(hidden_states + self.attn_dropout(attn_output))
        return self.layer_norm2(hidden_states + self.ffn_dropout(self.ffn(hidden_states)))


class DecoderBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, num_heads: int, context_length: int, dropout_rate: float
    ) -> None:
        super().__init__()
        self.causal_self_attention = MultiheadAttention(
            embedding_dim,
            num_heads,
            context_length,
            is_causal=True,
            dropout_rate=dropout_rate,
        )
        self.cross_attention = MultiheadAttention(
            embedding_dim,
            num_heads,
            context_length,
            is_crossattention=True,
            dropout_rate=dropout_rate,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.self_attn_dropout = nn.Dropout(dropout_rate)
        self.cross_attn_dropout = nn.Dropout(dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_padding_mask: torch.Tensor | None = None,
        encoder_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_output = self.causal_self_attention(
            hidden_states, padding_mask=decoder_padding_mask
        )
        hidden_states = self.layer_norm1(
            hidden_states + self.self_attn_dropout(attn_output)
        )
        attn_output = self.cross_attention(
            hidden_states,
            key_value_states=encoder_hidden_states,
            padding_mask=encoder_padding_mask,
        )
        hidden_states = self.layer_norm2(
            hidden_states + self.cross_attn_dropout(attn_output)
        )
        return self.layer_norm3(hidden_states + self.ffn_dropout(self.ffn(hidden_states)))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        num_layers: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderBlock(embedding_dim, num_heads, context_length, dropout_rate)
            for _ in range(num_layers)
        )

    def forward(
        self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, padding_mask=padding_mask)
        return hidden_states


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        num_layers: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderBlock(embedding_dim, num_heads, context_length, dropout_rate)
            for _ in range(num_layers)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_padding_mask: torch.Tensor | None = None,
        encoder_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states,
                decoder_padding_mask,
                encoder_padding_mask,
            )
        return hidden_states


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        num_layers: int,
        pad_token_id: int,
        dropout_rate: float,
        label_smoothing: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
        self.sqrt_d_model = math.sqrt(embedding_dim)
        self.token_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.position_encodings = SinusoidalPositionEmbedding(
            context_length, embedding_dim
        )
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.encoder = TransformerEncoder(
            embedding_dim, num_heads, context_length, num_layers, dropout_rate
        )
        self.decoder = TransformerDecoder(
            embedding_dim, num_heads, context_length, num_layers, dropout_rate
        )
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embeddings.weight
        with torch.no_grad():
            self.token_embeddings.weight[pad_token_id].zero_()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=module.embedding_dim**-0.5)

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        return self.embedding_dropout(
            self.token_embeddings(token_ids) * self.sqrt_d_model
            + self.position_encodings(positions)
        )

    def encode(
        self,
        source_ids: torch.Tensor,
        source_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(
            self._embed(source_ids), padding_mask=source_padding_mask
        )

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
        source_padding_mask: torch.Tensor,
        last_token_only: bool = False,
    ) -> torch.Tensor:
        decoder_hidden_states = self.decoder(
            self._embed(decoder_input_ids),
            encoder_hidden_states,
            decoder_padding_mask,
            source_padding_mask,
        )
        if last_token_only:
            decoder_hidden_states = decoder_hidden_states[:, -1:]
        return self.lm_head(decoder_hidden_states)

    def forward(
        self,
        source_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        source_padding_mask: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        last_token_only: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        encoder_hidden_states = self.encode(source_ids, source_padding_mask)
        logits = self.decode(
            decoder_input_ids,
            encoder_hidden_states,
            decoder_padding_mask,
            source_padding_mask,
            last_token_only=last_token_only,
        )
        loss = None
        if labels is not None:
            if last_token_only:
                raise ValueError("last_token_only cannot be used with labels")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=self.pad_token_id,
                label_smoothing=self.label_smoothing,
            )
        return {"logits": logits, "loss": loss}


def transformer_learning_rate(step: int, warmup_steps: int) -> float:
    step = max(step, 1)
    return EMBEDDING_DIM**-0.5 * min(
        step**-0.5, step * warmup_steps**-1.5
    )


def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()
    }


@torch.inference_mode()
def evaluate(
    model: Transformer,
    data_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    pad_token_id: int,
) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        tokens = batch["labels"].ne(pad_token_id).sum().item()
        with torch.autocast("cuda", dtype=amp_dtype):
            loss = model(**batch)["loss"]
        loss_sum += loss.item() * tokens
        token_count += tokens
    model.train()
    return loss_sum / max(token_count, 1)


def append_history(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(record, sort_keys=True) + "\n")


def save_checkpoint(
    path: Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    next_batch: int,
    update: int,
    best_validation_loss: float,
    elapsed_seconds: float,
    split_metadata: dict[str, object],
    data_config: dict[str, str],
    early_stopping_bad_checks: int,
    training_args: dict[str, object],
    best_checkpoints: list[dict[str, object]],
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "update": update,
        "best_validation_loss": best_validation_loss,
        "early_stopping_bad_checks": early_stopping_bad_checks,
        "elapsed_seconds": elapsed_seconds,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "python_rng_state": random.getstate(),
        "split": split_metadata,
        "data": data_config,
        "training_args": training_args,
        "best_checkpoints": best_checkpoints,
        "config": {
            "vocab_size": model.token_embeddings.num_embeddings,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "context_length": CONTEXT_LENGTH,
            "pad_token_id": model.pad_token_id,
            "dropout_rate": DROPOUT_RATE,
            "label_smoothing": LABEL_SMOOTHING,
        },
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def save_model_checkpoint(
    path: Path,
    model: Transformer,
    epoch: int,
    update: int,
    validation_loss: float,
    split_metadata: dict[str, object],
    data_config: dict[str, str],
    training_args: dict[str, object],
) -> None:
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "update": update,
        "best_validation_loss": validation_loss,
        "validation_loss": validation_loss,
        "split": split_metadata,
        "data": data_config,
        "training_args": training_args,
        "config": {
            "vocab_size": model.token_embeddings.num_embeddings,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "context_length": CONTEXT_LENGTH,
            "pad_token_id": model.pad_token_id,
            "dropout_rate": DROPOUT_RATE,
            "label_smoothing": LABEL_SMOOTHING,
        },
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def retain_best_checkpoint(
    checkpoint_dir: Path,
    model: Transformer,
    epoch: int,
    update: int,
    validation_loss: float,
    split_metadata: dict[str, object],
    data_config: dict[str, str],
    training_args: dict[str, object],
    best_checkpoints: list[dict[str, object]],
    keep_count: int,
) -> list[dict[str, object]]:
    ranked = list(best_checkpoints)
    qualifies = len(ranked) < keep_count or validation_loss < float(
        ranked[-1]["validation_loss"]
    )
    if not qualifies:
        return ranked
    filename = f"best_update_{update:08d}_loss_{validation_loss:.4f}.pt"
    save_model_checkpoint(
        checkpoint_dir / filename,
        model,
        epoch,
        update,
        validation_loss,
        split_metadata,
        data_config,
        training_args,
    )
    ranked.append(
        {
            "filename": filename,
            "update": update,
            "epoch": epoch,
            "validation_loss": validation_loss,
        }
    )
    ranked.sort(key=lambda item: float(item["validation_loss"]))
    while len(ranked) > keep_count:
        removed = ranked.pop()
        removed_path = checkpoint_dir / str(removed["filename"])
        if removed_path.exists():
            removed_path.unlink()
    return ranked


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve())
    data_dir = repo_root / "data/processed/wmt14_en_de"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-hours", type=float, default=8.0, help="Hours for this run")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--max-updates", type=int, default=None, help="Total update limit")
    parser.add_argument("--micro-batch-tokens", type=int, default=4_096)
    parser.add_argument("--max-sentences-per-batch", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--validation-interval", type=int, default=1_000)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--keep-best-checkpoints", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=4_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-source", type=Path, default=data_dir / "train.en"
    )
    parser.add_argument(
        "--train-target", type=Path, default=data_dir / "train.de"
    )
    parser.add_argument(
        "--validation-source", type=Path, default=data_dir / "validation.en"
    )
    parser.add_argument(
        "--validation-target", type=Path, default=data_dir / "validation.de"
    )
    parser.add_argument(
        "--test-source", type=Path, default=data_dir / "test.en"
    )
    parser.add_argument(
        "--test-target", type=Path, default=data_dir / "test.de"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=repo_root / "data/tokenizers/wmt14_bpe_en_de.json",
    )
    parser.add_argument(
        "--length-cache", type=Path, default=data_dir / "train.lengths.pt"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "checkpoints/transformer_en_de-v2",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_hours <= 0:
        raise ValueError("--max-hours must be positive")
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.micro_batch_tokens <= 0:
        raise ValueError("--micro-batch-tokens must be positive")
    if args.max_sentences_per_batch <= 0:
        raise ValueError("--max-sentences-per-batch must be positive")
    if args.validation_batch_size <= 0:
        raise ValueError("--validation-batch-size must be positive")
    if args.validation_interval <= 0:
        raise ValueError("--validation-interval must be positive")
    if args.early_stopping_patience <= 0:
        raise ValueError("--early-stopping-patience must be positive")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta cannot be negative")
    if args.keep_best_checkpoints <= 0:
        raise ValueError("--keep-best-checkpoints must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This training script expects a CUDA GPU")

    repo_root = find_repo_root(Path(__file__).resolve())
    def absolute(path: Path) -> Path:
        return path if path.is_absolute() else repo_root / path

    tokenizer_path = absolute(args.tokenizer_path)
    train_source_path = absolute(args.train_source)
    train_target_path = absolute(args.train_target)
    validation_source_path = absolute(args.validation_source)
    validation_target_path = absolute(args.validation_target)
    test_source_path = absolute(args.test_source)
    test_target_path = absolute(args.test_target)
    length_cache_path = absolute(args.length_cache)
    required_paths = (
        tokenizer_path,
        train_source_path,
        train_target_path,
        validation_source_path,
        validation_target_path,
        test_source_path,
        test_target_path,
        length_cache_path,
    )
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "WMT v2 inputs are missing:\n"
            + "\n".join(f"- {path}" for path in missing_paths)
            + "\nRun prepare_wmt14_en_de.py and train_wmt_tokenizer.py first."
        )

    checkpoint_dir = absolute(args.output_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = repo_root / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = checkpoint_dir / "latest.pt"
    best_checkpoint_path = checkpoint_dir / "best.pt"
    history_path = checkpoint_dir / "history.jsonl"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    if pad_token_id is None:
        raise ValueError("Tokenizer does not define [PAD]")
    train_dataset = IndexedParallelTextDataset(
        train_source_path, train_target_path, tokenizer, CONTEXT_LENGTH
    )
    validation_dataset = IndexedParallelTextDataset(
        validation_source_path,
        validation_target_path,
        tokenizer,
        CONTEXT_LENGTH,
    )
    length_cache = torch.load(length_cache_path, map_location="cpu", weights_only=True)
    tokenizer_digest = sha256(tokenizer.to_str().encode("utf-8")).hexdigest()
    if length_cache.get("tokenizer_sha256") != tokenizer_digest:
        raise ValueError("Token-length cache was built with a different tokenizer")
    source_lengths = length_cache["source_lengths"].to(torch.int32)
    target_lengths = length_cache["target_lengths"].to(torch.int32)
    if len(source_lengths) != len(train_dataset):
        raise ValueError("Token-length cache does not match the WMT training files")
    if length_cache.get("max_seq_len") != CONTEXT_LENGTH:
        raise ValueError("Token-length cache context length does not match the model")
    split_metadata: dict[str, object] = {
        "strategy": "external_validation_v1",
        "validation_examples": len(validation_dataset),
        "name": "newstest2013",
        "test_name": "newstest2014",
    }
    data_config = {
        "tokenizer_path": str(tokenizer_path),
        "train_source": str(train_source_path),
        "train_target": str(train_target_path),
        "validation_source": str(validation_source_path),
        "validation_target": str(validation_target_path),
        "test_source": str(test_source_path),
        "test_target": str(test_target_path),
        "length_cache": str(length_cache_path),
        "tokenizer_sha256": tokenizer_digest,
        "train_source_bytes": str(train_source_path.stat().st_size),
        "train_target_bytes": str(train_target_path.stat().st_size),
        "validation_source_bytes": str(validation_source_path.stat().st_size),
        "validation_target_bytes": str(validation_target_path.stat().st_size),
    }
    training_args: dict[str, object] = {
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "micro_batch_tokens": args.micro_batch_tokens,
        "max_sentences_per_batch": args.max_sentences_per_batch,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "validation_interval": args.validation_interval,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "keep_best_checkpoints": args.keep_best_checkpoints,
    }
    collator = TranslationCollator(pad_token_id)

    def make_train_loader(epoch_number: int) -> DataLoader:
        batch_sampler = TokenBucketBatchSampler(
            source_lengths,
            target_lengths,
            args.micro_batch_tokens,
            args.seed,
            epoch_number,
            max_sentences=args.max_sentences_per_batch,
        )
        return DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = Transformer(
        vocab_size,
        EMBEDDING_DIM,
        NUM_HEADS,
        CONTEXT_LENGTH,
        NUM_LAYERS,
        pad_token_id,
        DROPOUT_RATE,
        LABEL_SMOOTHING,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9
    )
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    epoch = next_batch = update = 0
    best_validation_loss = math.inf
    early_stopping_bad_checks = 0
    best_checkpoints: list[dict[str, object]] = []
    previous_elapsed = 0.0

    if args.resume and latest_checkpoint_path.exists():
        checkpoint = torch.load(
            latest_checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint_split = checkpoint.get("split")
        if checkpoint_split != split_metadata:
            raise ValueError(
                "Checkpoint validation data does not match the current WMT inputs; "
                "use matching paths or --no-resume with a new output directory."
            )
        if checkpoint.get("data") != data_config:
            raise ValueError(
                "Checkpoint data/tokenizer paths do not match the current v2 inputs"
            )
        saved_args = checkpoint.get("training_args", {})
        for name in (
            "seed",
            "warmup_steps",
            "micro_batch_tokens",
            "gradient_accumulation_steps",
        ):
            if saved_args.get(name) != training_args[name]:
                raise ValueError(
                    f"Checkpoint {name}={saved_args.get(name)!r} does not match "
                    f"current value {training_args[name]!r}"
                )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        epoch = checkpoint["epoch"]
        next_batch = checkpoint["next_batch"]
        update = checkpoint["update"]
        best_validation_loss = checkpoint["best_validation_loss"]
        early_stopping_bad_checks = checkpoint.get(
            "early_stopping_bad_checks", 0
        )
        best_checkpoints = checkpoint.get("best_checkpoints", [])
        previous_elapsed = checkpoint.get("elapsed_seconds", 0.0)
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        print(
            f"Resumed update {update:,}, epoch {epoch + 1}, batch {next_batch:,}",
            flush=True,
        )
    elif not args.resume:
        history_path.write_text("", encoding="utf-8")

    stop_requested = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            f"Received signal {signum}; checkpointing after the current optimizer update...",
            flush=True,
        )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print(
        f"Device: {torch.cuda.get_device_name(0)}; precision: {amp_dtype}; "
        f"train/validation: {len(train_dataset):,}/{len(validation_dataset):,}",
        flush=True,
    )
    print(
        f"Approximate effective batch ceiling: "
        f"{args.micro_batch_tokens * args.gradient_accumulation_steps:,} padded tokens; "
        f"session budget: {args.max_hours:g} h",
        flush=True,
    )

    started = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    last_batch = next_batch - 1
    accumulation_count = 0

    interval_loss_sum = 0.0
    interval_target_tokens = 0
    stop_reason = "unknown"
    while not stop and epoch < args.max_epochs:
        loader = make_train_loader(epoch)
        progress = tqdm(
            enumerate(loader), total=len(loader), desc=f"epoch {epoch + 1}"
        )
        token_count = 0
        for batch_index, batch in progress:
            if batch_index < next_batch:
                continue
            last_batch = batch_index
            batch = move_batch_to_device(batch, device)
            token_count += (
                batch["source_ids"].ne(pad_token_id).sum()
                + batch["labels"].ne(pad_token_id).sum()
            ).item()
            target_tokens = batch["labels"].ne(pad_token_id).sum().item()
            with torch.autocast("cuda", dtype=amp_dtype):
                batch_loss = model(**batch)["loss"]
                loss = batch_loss / args.gradient_accumulation_steps
            interval_loss_sum += batch_loss.item() * target_tokens
            interval_target_tokens += target_tokens
            scaler.scale(loss).backward()
            accumulation_count += 1
            if accumulation_count < args.gradient_accumulation_steps:
                continue
            accumulation_count = 0

            update += 1
            learning_rate = transformer_learning_rate(update, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            session_elapsed = time.monotonic() - started
            total_elapsed = previous_elapsed + session_elapsed

            if update % args.log_interval == 0:
                mean_training_loss = interval_loss_sum / max(
                    interval_target_tokens, 1
                )
                progress.set_postfix(
                    loss=f"{mean_training_loss:.3f}",
                    lr=f"{learning_rate:.2e}",
                    tokens=f"{token_count:,}",
                )
                append_history(
                    history_path,
                    {
                        "event": "train",
                        "epoch": epoch + 1,
                        "update": update,
                        "training_loss": mean_training_loss,
                        "learning_rate": learning_rate,
                        "elapsed_hours": total_elapsed / 3600,
                    },
                )
                interval_loss_sum = 0.0
                interval_target_tokens = 0
            token_count = 0

            if update % args.validation_interval == 0:
                validation_loss = evaluate(
                    model, validation_loader, device, amp_dtype, pad_token_id
                )
                print(
                    f"update {update:,} | val loss {validation_loss:.4f} | "
                    f"total {total_elapsed / 3600:.2f} h",
                    flush=True,
                )
                improved = (
                    validation_loss
                    < best_validation_loss - args.early_stopping_min_delta
                )
                best_checkpoints = retain_best_checkpoint(
                    checkpoint_dir,
                    model,
                    epoch,
                    update,
                    validation_loss,
                    split_metadata,
                    data_config,
                    training_args,
                    best_checkpoints,
                    args.keep_best_checkpoints,
                )
                append_history(
                    history_path,
                    {
                        "event": "validation",
                        "epoch": epoch + 1,
                        "update": update,
                        "validation_loss": validation_loss,
                        "best_validation_loss": (
                            validation_loss if improved else best_validation_loss
                        ),
                        "improved": improved,
                        "retained_best_checkpoints": best_checkpoints,
                        "elapsed_hours": total_elapsed / 3600,
                    },
                )
                if improved:
                    best_validation_loss = validation_loss
                    early_stopping_bad_checks = 0
                    save_checkpoint(
                        best_checkpoint_path,
                        model,
                        optimizer,
                        scaler,
                        epoch,
                        batch_index + 1,
                        update,
                        best_validation_loss,
                        total_elapsed,
                        split_metadata,
                        data_config,
                        early_stopping_bad_checks,
                        training_args,
                        best_checkpoints,
                    )
                else:
                    early_stopping_bad_checks += 1
                    print(
                        "No material validation improvement "
                        f"({early_stopping_bad_checks}/"
                        f"{args.early_stopping_patience})",
                        flush=True,
                    )
                # Persist the updated ranking immediately. Validation and regular
                # checkpoint intervals are configurable and need not coincide.
                save_checkpoint(
                    latest_checkpoint_path,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    batch_index + 1,
                    update,
                    best_validation_loss,
                    total_elapsed,
                    split_metadata,
                    data_config,
                    early_stopping_bad_checks,
                    training_args,
                    best_checkpoints,
                )
            if update % args.checkpoint_interval == 0:
                save_checkpoint(
                    latest_checkpoint_path,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    batch_index + 1,
                    update,
                    best_validation_loss,
                    total_elapsed,
                    split_metadata,
                    data_config,
                    early_stopping_bad_checks,
                    training_args,
                    best_checkpoints,
                )
                print(f"Saved checkpoint at update {update:,}", flush=True)

            reached_update_limit = (
                args.max_updates is not None and update >= args.max_updates
            )
            if stop_requested or reached_update_limit or session_elapsed >= args.max_hours * 3600:
                stop = True
                if stop_requested:
                    stop_reason = "signal"
                elif reached_update_limit:
                    stop_reason = "update limit"
                else:
                    stop_reason = "session time limit"
                break
            if early_stopping_bad_checks >= args.early_stopping_patience:
                stop = True
                stop_reason = "early stopping"
                break
        if not stop:
            epoch += 1
            next_batch = 0
    if not stop and epoch >= args.max_epochs:
        stop_reason = "epoch limit"

    total_elapsed = previous_elapsed + time.monotonic() - started
    save_checkpoint(
        latest_checkpoint_path,
        model,
        optimizer,
        scaler,
        epoch,
        last_batch + 1,
        update,
        best_validation_loss,
        total_elapsed,
        split_metadata,
        data_config,
        early_stopping_bad_checks,
        training_args,
        best_checkpoints,
    )
    print(
        f"Stopped safely ({stop_reason}) after "
        f"{total_elapsed / 3600:.2f} total hours and "
        f"{update:,} updates. Latest checkpoint: {latest_checkpoint_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
