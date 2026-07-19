"""Train a joint, boundary-aware byte-level BPE for WMT14 English-German."""

from __future__ import annotations

import argparse
from array import array
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
import torch
from tqdm.auto import tqdm

from train_transformer import find_repo_root


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]


def alternating_lines(source_path: Path, target_path: Path):
    with (
        source_path.open(encoding="utf-8") as source_file,
        target_path.open(encoding="utf-8") as target_file,
    ):
        for source, target in zip_longest(source_file, target_file):
            if source is None or target is None:
                raise ValueError("Source and target files have different line counts")
            source = source.strip()
            target = target.strip()
            if source and target:
                yield source
                yield target


def build_tokenizer(vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
    )
    return tokenizer


def create_length_cache(
    tokenizer: Tokenizer,
    source_path: Path,
    target_path: Path,
    output_path: Path,
    max_seq_len: int,
    batch_size: int = 4_096,
) -> None:
    source_lengths = array("H")
    target_lengths = array("H")
    source_batch: list[str] = []
    target_batch: list[str] = []

    def encode_pending() -> None:
        if not source_batch:
            return
        source_lengths.extend(
            min(len(encoding.ids), max_seq_len)
            for encoding in tokenizer.encode_batch(source_batch)
        )
        target_lengths.extend(
            min(len(encoding.ids), max_seq_len)
            for encoding in tokenizer.encode_batch(target_batch)
        )
        source_batch.clear()
        target_batch.clear()

    with (
        source_path.open(encoding="utf-8") as source_file,
        target_path.open(encoding="utf-8") as target_file,
    ):
        for source, target in tqdm(
            zip_longest(source_file, target_file), desc="token lengths"
        ):
            if source is None or target is None:
                raise ValueError("Source and target files have different line counts")
            source_batch.append(source.rstrip())
            target_batch.append(target.rstrip())
            if len(source_batch) >= batch_size:
                encode_pending()
        encode_pending()

    state = {
        "version": 1,
        "tokenizer_sha256": sha256(
            tokenizer.to_str().encode("utf-8")
        ).hexdigest(),
        "max_seq_len": max_seq_len,
        "source_lengths": torch.tensor(source_lengths, dtype=torch.int16),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.int16),
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(output_path)
    print(f"Saved {len(source_lengths):,} token lengths to {output_path}")


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve())
    data_dir = repo_root / "data/processed/wmt14_en_de"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=data_dir / "train.en")
    parser.add_argument("--target", type=Path, default=data_dir / "train.de")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data/tokenizers/wmt14_bpe_en_de.json",
    )
    parser.add_argument(
        "--length-cache",
        type=Path,
        default=data_dir / "train.lengths.pt",
    )
    parser.add_argument("--vocab-size", type=int, default=37_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument(
        "--skip-length-cache", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vocab_size <= len(SPECIAL_TOKENS):
        raise ValueError("--vocab-size is too small")
    if not args.source.exists() or not args.target.exists():
        raise FileNotFoundError(
            "Prepared WMT train files are missing; run prepare_wmt14_en_de.py first"
        )

    tokenizer = build_tokenizer(args.vocab_size)
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        alternating_lines(args.source, args.target),
        trainer=trainer,
    )
    if [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS] != [0, 1, 2, 3]:
        raise ValueError("Unexpected special-token ids")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))
    round_trip = "Ärgerst du dich über unvollständige Übersetzungen?"
    decoded = tokenizer.decode(
        tokenizer.encode(round_trip).ids, skip_special_tokens=True
    )
    if decoded != round_trip:
        raise ValueError(f"Tokenizer round-trip failed: {decoded!r}")
    print(f"Saved {tokenizer.get_vocab_size():,}-token vocabulary to {args.output}")
    print(f"Round trip: {decoded}")

    if not args.skip_length_cache:
        create_length_cache(
            tokenizer,
            args.source,
            args.target,
            args.length_cache,
            args.context_length,
        )


if __name__ == "__main__":
    main()
