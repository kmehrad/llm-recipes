import json
from pathlib import Path
import sys

from safetensors.torch import save_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
import torch


COMPONENT_DIR = (
    Path(__file__).parents[1] / "notebooks" / "00_transformer_components"
)
sys.path.insert(0, str(COMPONENT_DIR))

import train_transformer as modeling_transformer  # noqa: E402

sys.modules["modeling_transformer"] = modeling_transformer
from hub_inference import TransformerTranslator  # noqa: E402
from average_transformer_checkpoints import average_checkpoints  # noqa: E402
from evaluate_transformer import beam_search_generate  # noqa: E402
from prepare_wmt14_en_de import (  # noqa: E402
    acceptable_pair,
    normalized_text,
    parallel_lines,
    read_sgm_segments,
)
from train_transformer import (  # noqa: E402
    ParallelTextDataset,
    TokenBucketBatchSampler,
    Transformer,
    append_history,
    retain_best_checkpoint,
    source_grouped_split,
)
from train_wmt_tokenizer import (  # noqa: E402
    SPECIAL_TOKENS,
    build_tokenizer,
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


def test_wmt_sgm_and_filtering_helpers(tmp_path):
    sgm_path = tmp_path / "sample.sgm"
    sgm_path.write_text(
        '<seg id="1">Tom &amp; Mary</seg>\n<seg id="2">  Hello   world. </seg>\n'
    )

    assert read_sgm_segments(sgm_path) == ["Tom & Mary", "Hello world."]
    assert normalized_text("  A   B  ") == "A B"
    assert acceptable_pair("a short source", "eine kurze Übersetzung")
    assert not acceptable_pair("one", "eins zwei drei vier")


def test_wmt_parallel_lines_do_not_split_on_embedded_carriage_returns(tmp_path):
    source_path = tmp_path / "sample.en"
    target_path = tmp_path / "sample.de"
    source_path.write_bytes(b"first\rcontinued\nsecond\n")
    target_path.write_bytes(b"erste Zeile\nzweite Zeile\n")

    pairs = list(parallel_lines(source_path, target_path))

    assert pairs == [
        ("first\rcontinued\n", "erste Zeile\n"),
        ("second\n", "zweite Zeile\n"),
    ]


def test_boundary_aware_tokenizer_round_trips_unseen_german_characters():
    tokenizer = build_tokenizer(400)
    tokenizer.train_from_iterator(
        ["Ärgerst du dich?", "Do you get annoyed?", "unvollständige Übersetzungen"],
        trainer=BpeTrainer(
            vocab_size=400,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    text = "Ärgerst du dich über unvollständige Übersetzungen?"

    decoded = tokenizer.decode(
        tokenizer.encode(text).ids, skip_special_tokens=True
    )

    assert decoded == text


def test_token_bucket_batches_cover_every_example_within_budget():
    source_lengths = torch.tensor([4, 7, 3, 9, 5, 8])
    target_lengths = torch.tensor([5, 6, 4, 8, 5, 7])
    sampler = TokenBucketBatchSampler(
        source_lengths,
        target_lengths,
        max_tokens=30,
        seed=11,
        epoch=2,
        bucket_size=4,
        max_sentences=3,
    )

    batches = list(sampler)
    assert sorted(index for batch in batches for index in batch) == list(range(6))
    for batch in batches:
        padded_tokens = (
            max(int(source_lengths[index]) for index in batch)
            + max(int(target_lengths[index]) for index in batch)
        ) * len(batch)
        assert padded_tokens <= 30


def test_beam_search_returns_one_sequence_per_source():
    torch.manual_seed(0)
    model = Transformer(
        vocab_size=8,
        embedding_dim=8,
        num_heads=2,
        context_length=8,
        num_layers=1,
        pad_token_id=0,
        dropout_rate=0.0,
        label_smoothing=0.0,
    ).eval()
    source_ids = torch.tensor([[2, 4, 3], [2, 5, 3]])

    generated = beam_search_generate(
        model,
        source_ids,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        max_new_tokens=6,
        beam_size=4,
        length_penalty=0.6,
    )

    assert len(generated) == 2
    assert all(sequence[0] == 2 for sequence in generated)
    assert all(len(sequence) <= 6 for sequence in generated)


def test_checkpoint_averaging(tmp_path):
    paths = []
    for index, value in enumerate((1.0, 3.0), start=1):
        path = tmp_path / f"best-{index}.pt"
        torch.save(
            {
                "model": {
                    "weight": torch.tensor([value]),
                    "counter": torch.tensor([7]),
                },
                "config": {"name": "tiny"},
                "epoch": index,
                "update": index * 100,
                "validation_loss": 3.0 - index,
                "best_validation_loss": 3.0 - index,
            },
            path,
        )
        paths.append(path)

    averaged = average_checkpoints(paths)

    assert averaged["model"]["weight"].item() == 2.0
    assert averaged["model"]["counter"].item() == 7
    assert averaged["epoch"] == 2
    assert averaged["update"] == 200


def test_retains_only_checkpoints_with_lowest_validation_losses(tmp_path):
    model = Transformer(
        vocab_size=8,
        embedding_dim=8,
        num_heads=2,
        context_length=8,
        num_layers=1,
        pad_token_id=0,
        dropout_rate=0.0,
        label_smoothing=0.0,
    )
    retained = []
    for update, loss in enumerate((3.2, 3.1, 3.3, 3.0), start=1):
        retained = retain_best_checkpoint(
            tmp_path,
            model,
            epoch=0,
            update=update,
            validation_loss=loss,
            split_metadata={"name": "validation"},
            data_config={},
            training_args={},
            best_checkpoints=retained,
            keep_count=2,
        )

    assert [record["validation_loss"] for record in retained] == [3.0, 3.1]
    assert sorted(path.name for path in tmp_path.glob("best_update_*.pt")) == sorted(
        record["filename"] for record in retained
    )
