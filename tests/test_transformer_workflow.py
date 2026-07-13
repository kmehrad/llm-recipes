import json
from pathlib import Path
import sys

from safetensors.torch import save_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
import torch


COMPONENT_DIR = (
    Path(__file__).parents[1] / "notebooks" / "00_transformer_components"
)
sys.path.insert(0, str(COMPONENT_DIR))

import train_transformer as modeling_transformer  # noqa: E402

sys.modules["modeling_transformer"] = modeling_transformer
from hub_inference import TransformerTranslator  # noqa: E402
from train_transformer import (  # noqa: E402
    ParallelTextDataset,
    Transformer,
    append_history,
    source_grouped_split,
)


def test_source_grouped_split_is_deterministic_and_has_no_source_overlap():
    dataset = ParallelTextDataset.__new__(ParallelTextDataset)
    dataset.pairs = [
        ("same", "eins"),
        ("same", "zwei"),
        ("one", "eins"),
        ("two", "zwei"),
        ("three", "drei"),
        ("four", "vier"),
        ("five", "fünf"),
    ]

    train, validation, validation_indices = source_grouped_split(dataset, 3, seed=7)
    train_again, validation_again, validation_indices_again = source_grouped_split(
        dataset, 3, seed=7
    )

    assert validation_indices == validation_indices_again
    assert len(train) + len(validation) == len(dataset)
    assert set(train.indices).isdisjoint(validation.indices)
    assert set(train.indices) | set(validation.indices) == set(range(len(dataset)))
    train_sources = {dataset.pairs[index][0] for index in train.indices}
    validation_sources = {dataset.pairs[index][0] for index in validation.indices}
    assert train_sources.isdisjoint(validation_sources)
    assert train.indices == train_again.indices
    assert validation.indices == validation_again.indices


def test_exported_bundle_reloads_and_translates(tmp_path):
    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "hello": 4,
        "hallo": 5,
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", 2), ("[EOS]", 3)],
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))

    model_config = {
        "vocab_size": len(vocabulary),
        "embedding_dim": 8,
        "num_heads": 2,
        "context_length": 8,
        "num_layers": 1,
        "pad_token_id": 0,
        "dropout_rate": 0.0,
        "label_smoothing": 0.0,
    }
    torch.manual_seed(0)
    model = Transformer(**model_config).eval()
    save_model(model, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_config": model_config}), encoding="utf-8"
    )

    translator = TransformerTranslator.from_pretrained(tmp_path, device="cpu")
    result = translator.translate("hello", max_new_tokens=4)

    assert isinstance(result, str)


def test_history_is_machine_readable_jsonl(tmp_path):
    history_path = tmp_path / "history.jsonl"
    append_history(
        history_path,
        {
            "event": "validation",
            "update": 500,
            "validation_loss": 2.75,
            "improved": True,
        },
    )

    record = json.loads(history_path.read_text())
    assert record == {
        "event": "validation",
        "improved": True,
        "update": 500,
        "validation_loss": 2.75,
    }
