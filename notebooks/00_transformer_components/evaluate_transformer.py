"""Generate English-German validation predictions from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import random_split
from tqdm.auto import tqdm

from train_transformer import ParallelTextDataset, Transformer, find_repo_root


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "checkpoints/transformer_en_de/best.pt",
        help="Checkpoint to evaluate (default: best.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "evaluations/transformer_en_de/best_predictions.jsonl",
        help="Destination JSONL file",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Evaluate only this many validation examples",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum generated length including BOS (default: model context length)",
    )
    parser.add_argument(
        "--validation-examples",
        type=int,
        default=5_000,
        help="Must match the value used during training",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Must match the training split seed"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def encode_source(
    text: str,
    tokenizer: Tokenizer,
    context_length: int,
    eos_token_id: int,
) -> list[int]:
    token_ids = tokenizer.encode(text).ids
    if len(token_ids) > context_length:
        token_ids = token_ids[:context_length]
        token_ids[-1] = eos_token_id
    return token_ids


def pad_sequences(sequences: list[list[int]], pad_token_id: int) -> torch.Tensor:
    max_length = max(map(len, sequences))
    return torch.tensor(
        [
            sequence + [pad_token_id] * (max_length - len(sequence))
            for sequence in sequences
        ],
        dtype=torch.long,
    )


@torch.inference_mode()
def greedy_generate(
    model: Transformer,
    source_ids: torch.Tensor,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
) -> list[list[int]]:
    batch_size = source_ids.size(0)
    generated = torch.full(
        (batch_size, 1), bos_token_id, dtype=torch.long, device=source_ids.device
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=source_ids.device)
    source_padding_mask = source_ids.eq(pad_token_id)

    for _ in range(max_new_tokens - 1):
        outputs = model(
            source_ids=source_ids,
            decoder_input_ids=generated,
            source_padding_mask=source_padding_mask,
            decoder_padding_mask=generated.eq(pad_token_id),
        )
        next_tokens = outputs["logits"][:, -1].argmax(dim=-1)
        next_tokens = torch.where(finished, pad_token_id, next_tokens)
        generated = torch.cat((generated, next_tokens[:, None]), dim=1)
        finished |= next_tokens.eq(eos_token_id)
        if finished.all():
            break

    return generated.cpu().tolist()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    repo_root = find_repo_root(Path(__file__).resolve())
    tokenizer_path = repo_root / "data/tokenizers/tatoeba_bpe_en_de.json"
    english_path = repo_root / "data/raw/tatoeba_en_de/Tatoeba.de-en.en"
    german_path = repo_root / "data/raw/tatoeba_en_de/Tatoeba.de-en.de"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_token_id = tokenizer.token_to_id("[PAD]")
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    if None in (pad_token_id, bos_token_id, eos_token_id):
        raise ValueError("Tokenizer must define [PAD], [BOS], and [EOS]")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    config = checkpoint["config"]
    if config["vocab_size"] != tokenizer.get_vocab_size():
        raise ValueError("Checkpoint and tokenizer vocabulary sizes do not match")
    if config["pad_token_id"] != pad_token_id:
        raise ValueError("Checkpoint and tokenizer PAD ids do not match")

    device = resolve_device(args.device)
    model = Transformer(**config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    context_length = config["context_length"]
    max_new_tokens = args.max_new_tokens or context_length
    if not 2 <= max_new_tokens <= context_length:
        raise ValueError(
            f"--max-new-tokens must be between 2 and {context_length}"
        )

    all_dataset = ParallelTextDataset(
        english_path,
        german_path,
        tokenizer,
        context_length,
    )
    train_size = len(all_dataset) - args.validation_examples
    if train_size <= 0:
        raise ValueError("--validation-examples must be smaller than the dataset")
    _, validation_dataset = random_split(
        all_dataset,
        [train_size, args.validation_examples],
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_indices = validation_dataset.indices
    if args.max_examples is not None:
        validation_indices = validation_indices[: args.max_examples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    print(
        f"Checkpoint update {checkpoint['update']:,}; device {device}; "
        f"generating {len(validation_indices):,} predictions",
        flush=True,
    )

    try:
        with temporary_output.open("w", encoding="utf-8") as output_file:
            for start in tqdm(
                range(0, len(validation_indices), args.batch_size),
                desc="evaluating",
            ):
                batch_indices = validation_indices[start : start + args.batch_size]
                pairs = [all_dataset.pairs[index] for index in batch_indices]
                encoded_sources = [
                    encode_source(
                        source,
                        tokenizer,
                        context_length,
                        eos_token_id,
                    )
                    for source, _ in pairs
                ]
                source_ids = pad_sequences(encoded_sources, pad_token_id).to(device)
                generated_ids = greedy_generate(
                    model,
                    source_ids,
                    pad_token_id,
                    bos_token_id,
                    eos_token_id,
                    max_new_tokens,
                )

                for dataset_index, (source, target), prediction_ids in zip(
                    batch_indices, pairs, generated_ids, strict=True
                ):
                    record = {
                        "dataset_index": dataset_index,
                        "source": source,
                        "target": target,
                        "prediction": tokenizer.decode(
                            prediction_ids, skip_special_tokens=True
                        ),
                    }
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
    except KeyboardInterrupt:
        print(
            f"Evaluation interrupted. Partial results remain at {temporary_output}",
            flush=True,
        )
        raise SystemExit(130) from None

    temporary_output.replace(args.output)
    print(f"Wrote {len(validation_indices):,} predictions to {args.output}", flush=True)


if __name__ == "__main__":
    main()
