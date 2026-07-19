"""Interactively translate English text to German with the trained Transformer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from evaluate_transformer import beam_search_generate, encode_source
from train_transformer import Transformer, find_repo_root


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "checkpoints/transformer_en_de-v2/averaged.pt",
        help="Averaged or individual model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Tokenizer JSON (default: checkpoint metadata, then the WMT tokenizer)",
    )
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum output length including BOS (default: model context length)",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Translate one sentence and exit instead of starting the prompt",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def resolve_tokenizer_path(
    requested: Path | None, checkpoint: dict, repo_root: Path
) -> Path:
    if requested is not None:
        return requested
    data_config = checkpoint.get("data")
    if isinstance(data_config, dict) and data_config.get("tokenizer_path"):
        stored_path = Path(data_config["tokenizer_path"])
        if stored_path.exists():
            return stored_path
    return repo_root / "data/tokenizers/wmt14_bpe_en_de.json"


def main() -> None:
    args = parse_args()
    if args.beam_size <= 0:
        raise ValueError("--beam-size must be positive")
    if args.length_penalty < 0:
        raise ValueError("--length-penalty cannot be negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    repo_root = find_repo_root(Path(__file__).resolve())
    device = resolve_device(args.device)
    print(f"Loading {args.checkpoint} on {device} ...", flush=True)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    tokenizer_path = resolve_tokenizer_path(args.tokenizer, checkpoint, repo_root)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
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
    context_length = config["context_length"]
    max_new_tokens = args.max_new_tokens or context_length
    if not 2 <= max_new_tokens <= context_length:
        raise ValueError(
            f"--max-new-tokens must be between 2 and {context_length}"
        )

    model = Transformer(**config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    def translate(text: str) -> str:
        source_ids = torch.tensor(
            [encode_source(text, tokenizer, context_length, eos_token_id)],
            dtype=torch.long,
            device=device,
        )
        generated = beam_search_generate(
            model=model,
            source_ids=source_ids,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            amp_dtype=amp_dtype,
        )[0]
        return tokenizer.decode(generated, skip_special_tokens=True)

    if args.text is not None:
        if not args.text.strip():
            raise ValueError("Text to translate cannot be empty")
        print(translate(args.text))
        return

    print("Ready. Type English text and press Enter (Ctrl-D or Ctrl-C to quit).")
    try:
        while True:
            try:
                source = input("English> ").strip()
            except EOFError:
                print()
                break
            if not source:
                continue
            print(f"German>  {translate(source)}")
    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
