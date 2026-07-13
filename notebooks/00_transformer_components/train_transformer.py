"""Train the notebook's English-German Transformer outside Jupyter.

Run this script in tmux so training survives SSH and VS Code disconnects. It
automatically resumes from checkpoints/transformer_en_de/latest.pt by default.
"""

from __future__ import annotations

import argparse
from itertools import islice, zip_longest
import math
from pathlib import Path
import random
import signal
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm


EMBEDDING_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
CONTEXT_LENGTH = 128
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

    def forward(
        self,
        source_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        source_padding_mask: torch.Tensor,
        decoder_padding_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        encoder_hidden_states = self.encoder(
            self._embed(source_ids), padding_mask=source_padding_mask
        )
        decoder_hidden_states = self.decoder(
            self._embed(decoder_input_ids),
            encoder_hidden_states,
            decoder_padding_mask,
            source_padding_mask,
        )
        logits = self.lm_head(decoder_hidden_states)
        loss = None
        if labels is not None:
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
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "update": update,
        "best_validation_loss": best_validation_loss,
        "elapsed_seconds": elapsed_seconds,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "python_rng_state": random.getstate(),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-hours", type=float, default=8.0, help="Hours for this run")
    parser.add_argument("--max-updates", type=int, default=None, help="Total update limit")
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--validation-examples", type=int, default=5_000)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=4_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_hours <= 0:
        raise ValueError("--max-hours must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This training script expects a CUDA GPU")

    repo_root = find_repo_root(Path(__file__).resolve())
    tokenizer_path = repo_root / "data/tokenizers/tatoeba_bpe_en_de.json"
    english_path = repo_root / "data/raw/tatoeba_en_de/Tatoeba.de-en.en"
    german_path = repo_root / "data/raw/tatoeba_en_de/Tatoeba.de-en.de"
    checkpoint_dir = repo_root / "checkpoints/transformer_en_de"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = checkpoint_dir / "latest.pt"
    best_checkpoint_path = checkpoint_dir / "best.pt"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    if pad_token_id is None:
        raise ValueError("Tokenizer does not define [PAD]")
    all_dataset = ParallelTextDataset(
        english_path, german_path, tokenizer, CONTEXT_LENGTH
    )
    train_size = len(all_dataset) - args.validation_examples
    if train_size <= 0:
        raise ValueError("--validation-examples must be smaller than the dataset")
    train_dataset, validation_dataset = random_split(
        all_dataset,
        [train_size, args.validation_examples],
        generator=torch.Generator().manual_seed(args.seed),
    )
    collator = TranslationCollator(pad_token_id)

    def make_train_loader(epoch_number: int) -> DataLoader:
        return DataLoader(
            train_dataset,
            batch_size=args.micro_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + epoch_number),
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            drop_last=True,
        )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.micro_batch_size,
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
    previous_elapsed = 0.0

    if args.resume and latest_checkpoint_path.exists():
        checkpoint = torch.load(
            latest_checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        epoch = checkpoint["epoch"]
        next_batch = checkpoint["next_batch"]
        update = checkpoint["update"]
        best_validation_loss = checkpoint["best_validation_loss"]
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
        f"Effective batch: {args.micro_batch_size * args.gradient_accumulation_steps} pairs; "
        f"session budget: {args.max_hours:g} h",
        flush=True,
    )

    started = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    last_batch = next_batch - 1
    accumulation_count = 0

    while not stop:
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
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = model(**batch)["loss"] / args.gradient_accumulation_steps
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
                progress.set_postfix(
                    loss=f"{loss.item() * args.gradient_accumulation_steps:.3f}",
                    lr=f"{learning_rate:.2e}",
                    tokens=f"{token_count:,}",
                )
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
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
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
                )
                print(f"Saved checkpoint at update {update:,}", flush=True)

            reached_update_limit = (
                args.max_updates is not None and update >= args.max_updates
            )
            if stop_requested or reached_update_limit or session_elapsed >= args.max_hours * 3600:
                stop = True
                break
        if not stop:
            epoch += 1
            next_batch = 0

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
    )
    print(
        f"Stopped safely after {total_elapsed / 3600:.2f} total hours and "
        f"{update:,} updates. Latest checkpoint: {latest_checkpoint_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
