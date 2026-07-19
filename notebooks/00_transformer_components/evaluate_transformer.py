"""Decode WMT English-German data and report sacreBLEU and chrF++."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF
import torch
import torch.nn.functional as F
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
        default=repo_root / "checkpoints/transformer_en_de/averaged.pt",
        help="Averaged or individual model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "evaluations/transformer_en_de/newstest2014_beam4.jsonl",
        help="Destination JSONL file",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="External WMT split to evaluate",
    )
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
        help="Legacy checkpoints only: must match training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Legacy checkpoints only: must match the training split seed",
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
def beam_search_generate(
    model: Transformer,
    source_ids: torch.Tensor,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    beam_size: int,
    length_penalty: float,
    amp_dtype: torch.dtype | None = None,
) -> list[list[int]]:
    batch_size = source_ids.size(0)
    source_padding_mask = source_ids.eq(pad_token_id)
    with torch.autocast(
        source_ids.device.type,
        dtype=amp_dtype,
        enabled=amp_dtype is not None,
    ):
        encoder_hidden_states = model.encode(source_ids, source_padding_mask)
    encoder_hidden_states = encoder_hidden_states.repeat_interleave(
        beam_size, dim=0
    )
    expanded_source_mask = source_padding_mask.repeat_interleave(
        beam_size, dim=0
    )
    generated = torch.full(
        (batch_size, beam_size, 1),
        bos_token_id,
        dtype=torch.long,
        device=source_ids.device,
    )
    scores = torch.full(
        (batch_size, beam_size),
        float("-inf"),
        device=source_ids.device,
    )
    scores[:, 0] = 0.0
    lengths = torch.zeros_like(scores, dtype=torch.long)
    finished = torch.zeros_like(scores, dtype=torch.bool)
    vocab_size = model.token_embeddings.num_embeddings

    for _ in range(max_new_tokens - 1):
        flat_generated = generated.view(batch_size * beam_size, -1)
        with torch.autocast(
            source_ids.device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            logits = model.decode(
                flat_generated,
                encoder_hidden_states,
                flat_generated.eq(pad_token_id),
                expanded_source_mask,
                last_token_only=True,
            )
        log_probabilities = F.log_softmax(
            logits[:, -1].float(), dim=-1
        ).view(batch_size, beam_size, vocab_size)
        if finished.any():
            log_probabilities = log_probabilities.masked_fill(
                finished[:, :, None], float("-inf")
            )
            log_probabilities[:, :, pad_token_id] = torch.where(
                finished,
                torch.zeros_like(scores),
                log_probabilities[:, :, pad_token_id],
            )
        candidate_scores = scores[:, :, None] + log_probabilities
        scores, candidate_indices = candidate_scores.view(batch_size, -1).topk(
            beam_size, dim=-1
        )
        parent_beams = candidate_indices // vocab_size
        next_tokens = candidate_indices % vocab_size
        generated = torch.gather(
            generated,
            1,
            parent_beams[:, :, None].expand(-1, -1, generated.size(-1)),
        )
        previous_finished = torch.gather(finished, 1, parent_beams)
        lengths = torch.gather(lengths, 1, parent_beams)
        generated = torch.cat((generated, next_tokens[:, :, None]), dim=-1)
        lengths = lengths + (~previous_finished).long()
        finished = previous_finished | next_tokens.eq(eos_token_id)
        if finished.all():
            break

    penalties = ((5.0 + lengths.clamp_min(1).float()) / 6.0) ** length_penalty
    best_beams = (scores / penalties).argmax(dim=-1)
    selected = generated[
        torch.arange(batch_size, device=generated.device), best_beams
    ]
    return selected.cpu().tolist()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.beam_size <= 0:
        raise ValueError("--beam-size must be positive")
    if args.length_penalty < 0:
        raise ValueError("--length-penalty cannot be negative")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    repo_root = find_repo_root(Path(__file__).resolve())
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    data_config = checkpoint.get("data")
    split = checkpoint.get("split")
    external_data = (
        isinstance(data_config, dict)
        and split is not None
        and split.get("strategy") == "external_validation_v1"
    )
    if external_data:
        tokenizer_path = Path(data_config["tokenizer_path"])
        source_key = f"{args.split}_source"
        target_key = f"{args.split}_target"
        source_path = Path(data_config[source_key])
        target_path = Path(data_config[target_key])
    else:
        tokenizer_path = repo_root / "data/tokenizers/tatoeba_bpe_en_de.json"
        source_path = repo_root / "data/raw/tatoeba_en_de/Tatoeba.de-en.en"
        target_path = repo_root / "data/raw/tatoeba_en_de/Tatoeba.de-en.de"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_token_id = tokenizer.token_to_id("[PAD]")
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    if None in (pad_token_id, bos_token_id, eos_token_id):
        raise ValueError("Tokenizer must define [PAD], [BOS], and [EOS]")

    config = checkpoint["config"]
    if config["vocab_size"] != tokenizer.get_vocab_size():
        raise ValueError("Checkpoint and tokenizer vocabulary sizes do not match")
    if config["pad_token_id"] != pad_token_id:
        raise ValueError("Checkpoint and tokenizer PAD ids do not match")

    device = resolve_device(args.device)
    model = Transformer(**config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    context_length = config["context_length"]
    max_new_tokens = args.max_new_tokens or context_length
    if not 2 <= max_new_tokens <= context_length:
        raise ValueError(
            f"--max-new-tokens must be between 2 and {context_length}"
        )

    all_dataset = ParallelTextDataset(
        source_path,
        target_path,
        tokenizer,
        context_length,
    )
    if external_data:
        validation_indices = list(range(len(all_dataset)))
        print(
            f"Using WMT {args.split} split with "
            f"{len(validation_indices):,} examples",
            flush=True,
        )
    elif split is not None:
        validation_indices = split.get("validation_indices")
        if not isinstance(validation_indices, list) or not validation_indices:
            raise ValueError("Checkpoint split metadata has no validation indices")
        if min(validation_indices) < 0 or max(validation_indices) >= len(all_dataset):
            raise ValueError("Checkpoint validation indices exceed the current dataset")
        print(
            f"Using checkpoint split {split.get('strategy')} with "
            f"{len(validation_indices):,} validation examples",
            flush=True,
        )
    else:
        train_size = len(all_dataset) - args.validation_examples
        if train_size <= 0:
            raise ValueError("--validation-examples must be smaller than the dataset")
        _, validation_dataset = random_split(
            all_dataset,
            [train_size, args.validation_examples],
            generator=torch.Generator().manual_seed(args.seed),
        )
        validation_indices = validation_dataset.indices
        print(
            "Checkpoint predates stored split metadata; reconstructed the legacy "
            "row-random split from --seed and --validation-examples.",
            flush=True,
        )
    if args.max_examples is not None:
        validation_indices = validation_indices[: args.max_examples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    print(
        f"Checkpoint update {checkpoint['update']:,}; device {device}; "
        f"generating {len(validation_indices):,} predictions with "
        f"beam={args.beam_size}, length penalty={args.length_penalty:g}",
        flush=True,
    )

    hypotheses: list[str] = []
    references: list[str] = []
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
                generated_ids = beam_search_generate(
                    model,
                    source_ids,
                    pad_token_id,
                    bos_token_id,
                    eos_token_id,
                    max_new_tokens,
                    args.beam_size,
                    args.length_penalty,
                    amp_dtype,
                )

                for dataset_index, (source, target), prediction_ids in zip(
                    batch_indices, pairs, generated_ids, strict=True
                ):
                    prediction = tokenizer.decode(
                        prediction_ids, skip_special_tokens=True
                    )
                    record = {
                        "dataset_index": dataset_index,
                        "source": source,
                        "target": target,
                        "prediction": prediction,
                    }
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    hypotheses.append(prediction)
                    references.append(target)
                output_file.flush()
    except KeyboardInterrupt:
        print(
            f"Evaluation interrupted. Partial results remain at {temporary_output}",
            flush=True,
        )
        raise SystemExit(130) from None

    temporary_output.replace(args.output)
    bleu_metric = BLEU()
    chrf_metric = CHRF(word_order=2)
    bleu = bleu_metric.corpus_score(hypotheses, [references])
    chrf = chrf_metric.corpus_score(hypotheses, [references])
    metrics = {
        "checkpoint": str(args.checkpoint),
        "split": args.split if external_data else "legacy_validation",
        "examples": len(hypotheses),
        "beam_size": args.beam_size,
        "length_penalty": args.length_penalty,
        "bleu": bleu.score,
        "bleu_signature": str(bleu_metric.get_signature()),
        "chrf2": chrf.score,
        "chrf_signature": str(chrf_metric.get_signature()),
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(validation_indices):,} predictions to {args.output}", flush=True)
    print(f"BLEU = {bleu.score:.2f}; chrF++ = {chrf.score:.2f}", flush=True)
    print(f"Wrote metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
