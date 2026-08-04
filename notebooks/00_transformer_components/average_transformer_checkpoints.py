"""Average the retained best Transformer checkpoints for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_transformer import find_repo_root


def average_checkpoints(paths: list[Path]) -> dict[str, object]:
    if not paths:
        raise ValueError("No checkpoints were selected")
    template: dict[str, object] | None = None
    averaged: dict[str, torch.Tensor] = {}
    source_records = []

    for index, path in enumerate(paths):
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
        if template is None:
            template = checkpoint
        elif checkpoint["config"] != template["config"]:
            raise ValueError(f"Model configuration differs in {path}")
        source_records.append(
            {
                "path": str(path),
                "update": checkpoint["update"],
                "epoch": checkpoint["epoch"],
                "validation_loss": checkpoint.get(
                    "validation_loss", checkpoint["best_validation_loss"]
                ),
            }
        )
        for name, tensor in checkpoint["model"].items():
            if tensor.is_floating_point():
                value = tensor.float()
                if index == 0:
                    averaged[name] = value.clone()
                else:
                    averaged[name].add_(value)
            elif index == 0:
                averaged[name] = tensor.clone()
            elif not torch.equal(averaged[name], tensor):
                raise ValueError(f"Non-floating buffer {name} differs in {path}")
        del checkpoint

    for tensor in averaged.values():
        if tensor.is_floating_point():
            tensor.div_(len(paths))
    return {
        "model": averaged,
        "config": template["config"],
        "split": template.get("split"),
        "data": template.get("data"),
        "training_args": template.get("training_args"),
        "epoch": max(record["epoch"] for record in source_records),
        "update": max(record["update"] for record in source_records),
        "best_validation_loss": min(
            record["validation_loss"] for record in source_records
        ),
        "averaged_checkpoints": source_records,
    }


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve())
    checkpoint_dir = repo_root / "checkpoints/transformer_en_de"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-checkpoint",
        type=Path,
        default=checkpoint_dir / "latest.pt",
        help="latest.pt containing the ranked best-checkpoint manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=checkpoint_dir / "averaged.pt",
    )
    parser.add_argument("--count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    manifest = torch.load(
        args.training_checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    records = manifest.get("best_checkpoints", [])
    if len(records) < args.count:
        raise ValueError(
            f"Requested {args.count} checkpoints, but only {len(records)} are retained"
        )
    checkpoint_dir = args.training_checkpoint.parent
    paths = [
        checkpoint_dir / str(record["filename"])
        for record in records[: args.count]
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Retained checkpoints are missing: {missing}")

    averaged = average_checkpoints(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(averaged, temporary_path)
    temporary_path.replace(args.output)
    print("Averaged checkpoints:")
    for record in averaged["averaged_checkpoints"]:
        print(
            f"- update {record['update']:,}, "
            f"validation loss {record['validation_loss']:.4f}"
        )
    print(f"Saved averaged model to {args.output}")


if __name__ == "__main__":
    main()
