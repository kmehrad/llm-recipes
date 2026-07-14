"""Download and prepare the WMT14 English-German corpus used by Transformer.

The recipe combines Europarl v7, Common Crawl, and News Commentary v9, uses
newstest2013 for validation, and newstest2014 for the final test set.
"""

from __future__ import annotations

import argparse
from hashlib import blake2b
import html
import json
from pathlib import Path
import re
import tarfile
from urllib.request import urlretrieve

from tqdm.auto import tqdm

from train_transformer import find_repo_root


ARCHIVES = {
    "training-parallel-europarl-v7.tgz": (
        "https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    ),
    "training-parallel-commoncrawl.tgz": (
        "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    ),
    "training-parallel-nc-v9.tgz": (
        "https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz"
    ),
    "dev.tgz": "https://www.statmt.org/wmt14/dev.tgz",
    "test-full.tgz": "https://www.statmt.org/wmt14/test-full.tgz",
}

TRAINING_CORPORA = (
    "europarl-v7.de-en",
    "commoncrawl.de-en",
    "news-commentary-v9.de-en",
)
SEGMENT_PATTERN = re.compile(r"""<seg\s+id=["'][^"']+["']>(.*?)</seg>""")


def normalized_text(text: str) -> str:
    return " ".join(html.unescape(text).split())


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination, filter="data")


def download_archives(download_dir: Path) -> list[Path]:
    download_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, url in ARCHIVES.items():
        destination = download_dir / filename
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            print(f"Downloading {url}", flush=True)
            urlretrieve(url, temporary)
            temporary.replace(destination)
        paths.append(destination)
    return paths


def find_unique(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {filename!r} under {root}, found {len(matches)}"
        )
    return matches[0]


def read_sgm_segments(path: Path) -> list[str]:
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SEGMENT_PATTERN.search(line)
        if match:
            segments.append(normalized_text(match.group(1)))
    if not segments:
        raise ValueError(f"No <seg> records found in {path}")
    return segments


def write_parallel(
    sources: list[str],
    targets: list[str],
    source_path: Path,
    target_path: Path,
) -> None:
    if len(sources) != len(targets):
        raise ValueError("Source and target segment counts differ")
    source_path.write_text("\n".join(sources) + "\n", encoding="utf-8")
    target_path.write_text("\n".join(targets) + "\n", encoding="utf-8")


def pair_digest(source: str, target: str) -> bytes:
    value = source.encode() + b"\0" + target.encode()
    return blake2b(value, digest_size=16).digest()


def acceptable_pair(source: str, target: str) -> bool:
    source_words = len(source.split())
    target_words = len(target.split())
    if not source_words or not target_words:
        return False
    if source_words > 250 or target_words > 250:
        return False
    ratio = max(source_words / target_words, target_words / source_words)
    return ratio <= 3.0 and len(source) <= 2_000 and len(target) <= 2_000


def parallel_lines(source_path: Path, target_path: Path):
    """Yield aligned records, treating only LF as a corpus record separator.

    News Commentary contains bare carriage returns inside some records. Python's
    default universal-newline mode treats those as new lines, but their counts
    differ by language and therefore break otherwise aligned parallel data.
    """
    with (
        source_path.open(encoding="utf-8", newline="\n") as source_file,
        target_path.open(encoding="utf-8", newline="\n") as target_file,
    ):
        yield from zip(source_file, target_file, strict=True)


def prepare_training_data(
    extracted_dir: Path,
    output_source: Path,
    output_target: Path,
    excluded_pairs: set[bytes],
    max_training_pairs: int | None,
) -> dict[str, int]:
    seen = set(excluded_pairs)
    counts = {"read": 0, "written": 0, "filtered": 0, "duplicate": 0}
    with (
        output_source.open("w", encoding="utf-8") as source_output,
        output_target.open("w", encoding="utf-8") as target_output,
    ):
        for corpus in TRAINING_CORPORA:
            source_path = find_unique(extracted_dir, f"{corpus}.en")
            target_path = find_unique(extracted_dir, f"{corpus}.de")
            iterator = parallel_lines(source_path, target_path)
            for source_line, target_line in tqdm(iterator, desc=corpus):
                counts["read"] += 1
                source = normalized_text(source_line)
                target = normalized_text(target_line)
                if not acceptable_pair(source, target):
                    counts["filtered"] += 1
                    continue
                digest = pair_digest(source, target)
                if digest in seen:
                    counts["duplicate"] += 1
                    continue
                seen.add(digest)
                source_output.write(source + "\n")
                target_output.write(target + "\n")
                counts["written"] += 1
                if (
                    max_training_pairs is not None
                    and counts["written"] >= max_training_pairs
                ):
                    return counts
    return counts


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=repo_root / "data/downloads/wmt14_en_de",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data/processed/wmt14_en_de",
    )
    parser.add_argument(
        "--max-training-pairs",
        type=int,
        default=None,
        help="Optional smoke-test limit after filtering",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_training_pairs is not None and args.max_training_pairs <= 0:
        raise ValueError("--max-training-pairs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = args.download_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for archive_path in download_archives(args.download_dir):
        marker = extracted_dir / f".{archive_path.name}.extracted"
        if not marker.exists():
            print(f"Extracting {archive_path.name}", flush=True)
            safe_extract(archive_path, extracted_dir)
            marker.touch()

    validation_source = read_sgm_segments(
        find_unique(extracted_dir, "newstest2013-src.en.sgm")
    )
    validation_target = read_sgm_segments(
        find_unique(extracted_dir, "newstest2013-ref.de.sgm")
    )
    test_source = read_sgm_segments(
        find_unique(extracted_dir, "newstest2014-deen-src.en.sgm")
    )
    test_target = read_sgm_segments(
        find_unique(extracted_dir, "newstest2014-deen-ref.de.sgm")
    )
    write_parallel(
        validation_source,
        validation_target,
        args.output_dir / "validation.en",
        args.output_dir / "validation.de",
    )
    write_parallel(
        test_source,
        test_target,
        args.output_dir / "test.en",
        args.output_dir / "test.de",
    )

    excluded_pairs = {
        pair_digest(source, target)
        for source, target in zip(
            validation_source + test_source,
            validation_target + test_target,
            strict=True,
        )
    }
    counts = prepare_training_data(
        extracted_dir,
        args.output_dir / "train.en",
        args.output_dir / "train.de",
        excluded_pairs,
        args.max_training_pairs,
    )
    metadata = {
        "recipe": "wmt14_en_de_transformer_v1",
        "training_corpora": list(TRAINING_CORPORA),
        "validation": "newstest2013",
        "test": "newstest2014",
        "archive_urls": ARCHIVES,
        "counts": {
            **counts,
            "validation": len(validation_source),
            "test": len(test_source),
        },
        "filter": {
            "max_words": 250,
            "max_word_ratio": 3.0,
            "max_characters": 2_000,
            "exact_pair_deduplication": True,
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata["counts"], indent=2), flush=True)
    print(f"Prepared WMT14 data under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
