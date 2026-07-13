# Transformer component notebooks

The notebooks in this directory build an English-to-German Transformer from
its individual components. Use `4_transformer.ipynb` interactively to inspect
the complete model and verify that training works. Its training loop defaults
to a single optimizer update (four micro-batches) through these flags:

```python
notebook_smoke_test = True
notebook_max_updates = 1
```

Keep the smoke-test flag enabled for normal notebook use. Long training runs
belong in `train_transformer.py`, which is independent of the notebook kernel
and automatically resumes from its latest checkpoint.

The notebook keeps validation disabled by default and writes its one-update
checkpoint to `checkpoints/transformer_en_de_notebook_smoke_test/`, so it
cannot overwrite a full training run. Hub publishing lives only in
`5_publish_transformer_to_hf.ipynb`.

## Run training over SSH

From the repository root, create a named `tmux` session:

```bash
tmux new -s transformer
```

Inside that session, start a clean rerun in a new directory and save the
terminal output to a persistent log:

```bash
mkdir -p checkpoints/transformer_en_de
uv run python notebooks/00_transformer_components/train_transformer.py \
  --no-resume \
  --output-dir checkpoints/transformer_en_de \
  --max-epochs 200 \
  2>&1 | tee -a checkpoints/transformer_en_de/training.log
```

This preserves the previous 200-epoch run under
`checkpoints/transformer_en_de/`. The rerun can reach 200 epochs, but normally
stops earlier when validation loss has not materially improved for 10
consecutive validation checks.

The full corpus has 322,406 usable sentence pairs. Training holds out 5,000
pairs and uses every remaining pair. The split keeps all rows with the same
English source sentence together, preventing alternate translations of one
source from leaking across train and validation sets.

Detach from `tmux` without stopping training by pressing `Ctrl-b`, releasing
the keys, and then pressing `d`. It is safe for SSH or VS Code to disconnect
after that. Reconnect later with:

```bash
tmux attach -t transformer
```

List existing sessions with `tmux ls`. If `tmux` is not installed on the
workstation, install it first or ask the system administrator to do so.

## Checkpoints and resume

Training writes atomic checkpoints under:

```text
checkpoints/transformer_en_de/latest.pt
checkpoints/transformer_en_de/best.pt
checkpoints/transformer_en_de/history.jsonl
```

`latest.pt` is saved every 100 optimizer updates and whenever the process
receives `SIGINT` or `SIGTERM`. Starting the same command again resumes from
`latest.pt` automatically when `--no-resume` is removed. Each invocation receives a fresh eight-hour time
budget, so an existing checkpoint does not cause a resumed run to stop
immediately.

To deliberately start over without loading `latest.pt`, pass `--no-resume`.
This does not delete existing files; the new run will overwrite `latest.pt`
when it reaches a checkpoint.

Legacy checkpoints used a row-random validation split and cannot be resumed by
the grouped-split trainer. They remain evaluable, but a new grouped run must
use `--no-resume` or a new `--output-dir`.

## Useful options

Show every option:

```bash
uv run python notebooks/00_transformer_components/train_transformer.py --help
```

Run for six hours:

```bash
uv run python notebooks/00_transformer_components/train_transformer.py \
  --output-dir checkpoints/transformer_en_de \
  --max-hours 6
```

Stop at a total of 10,000 optimizer updates, including resumed updates:

```bash
uv run python notebooks/00_transformer_components/train_transformer.py \
  --output-dir checkpoints/transformer_en_de \
  --max-hours 24 \
  --max-updates 10000
```

Reduce GPU memory usage by lowering the micro-batch size while increasing
gradient accumulation to retain an effective batch of 256 sentence pairs:

```bash
uv run python notebooks/00_transformer_components/train_transformer.py \
  --output-dir checkpoints/transformer_en_de \
  --micro-batch-size 32 --gradient-accumulation-steps 8
```

The most important defaults are:

- micro-batch size: 64
- gradient accumulation: 4
- validation interval: 500 optimizer updates
- early-stopping patience: 10 validation checks
- minimum validation improvement: 0.001
- maximum epochs: 200
- checkpoint interval: 100 optimizer updates
- per-invocation time budget: 8 hours
- automatic resume: enabled

Use `Ctrl-c` once when you want to stop manually. The script finishes the
current optimizer update, saves `latest.pt`, and exits cleanly.

## Generate validation predictions

Stop the training process before evaluation so both models do not compete for
GPU memory. The evaluation script loads `best.pt` by default, reconstructs the
validation indices stored inside the checkpoint and performs batched greedy
decoding:

```bash
uv run python notebooks/00_transformer_components/evaluate_transformer.py
```

The completed output is written to:

```text
evaluations/transformer_en_de/best_predictions.jsonl
```

Each line is a UTF-8 JSON object with the original dataset index, English
source, German ground truth, and generated German prediction:

```json
{"dataset_index": 123, "source": "How are you?", "target": "Wie geht es dir?", "prediction": "Wie geht es dir?"}
```

Run a quick 20-example check before evaluating the complete split:

```bash
uv run python notebooks/00_transformer_components/evaluate_transformer.py \
  --max-examples 20 \
  --output evaluations/transformer_en_de/quick_predictions.jsonl
```

New checkpoints carry their exact validation indices, so evaluation does not
need split arguments. The old `--seed` and `--validation-examples` options
remain as a fallback for legacy checkpoints. Use `--checkpoint` to evaluate
another checkpoint and `--batch-size` to adjust GPU memory consumption. The
script writes to a `.tmp` file while generation is in progress and renames it
to the requested output only after all examples finish. If interrupted, the
partial `.tmp` file is retained.

## Review and optionally publish to Hugging Face

Publishing is deliberately separate from training. Open
`5_publish_transformer_to_hf.ipynb` after evaluating `best.pt`. It displays
the checkpoint metrics and saved translation samples, exports a self-contained
Safetensors bundle, and reloads that bundle locally.

The notebook defaults to `publish_to_hub = False`. If the metrics and sample
translations are satisfactory, authenticate outside the notebook with
`hf auth login` or `HF_TOKEN`, change the flag to `True`, and run only the
upload cell. The exported repository includes weights, tokenizer, config,
model/inference code, and a model card.
