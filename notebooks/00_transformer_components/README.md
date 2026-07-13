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

The notebook keeps validation and Hub upload disabled by default and writes its
one-update checkpoint to `checkpoints/transformer_en_de_notebook_smoke_test/`,
so it cannot overwrite a full training run.

## Run training over SSH

From the repository root, create a named `tmux` session:

```bash
tmux new -s transformer
```

Inside that session, start an eight-hour training run and save the terminal
output to a persistent log:

```bash
mkdir -p checkpoints/transformer_en_de
uv run python notebooks/00_transformer_components/train_transformer.py \
  2>&1 | tee -a checkpoints/transformer_en_de/training.log
```

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
```

`latest.pt` is saved every 100 optimizer updates and whenever the process
receives `SIGINT` or `SIGTERM`. Starting the same command again resumes from
`latest.pt` automatically. Each invocation receives a fresh eight-hour time
budget, so an existing checkpoint does not cause a resumed run to stop
immediately.

To deliberately start over without loading `latest.pt`, pass `--no-resume`.
This does not delete existing files; the new run will overwrite `latest.pt`
when it reaches a checkpoint.

## Useful options

Show every option:

```bash
uv run python notebooks/00_transformer_components/train_transformer.py --help
```

Run for six hours:

```bash
uv run python notebooks/00_transformer_components/train_transformer.py --max-hours 6
```

Stop at a total of 10,000 optimizer updates, including resumed updates:

```bash
uv run python notebooks/00_transformer_components/train_transformer.py \
  --max-hours 24 --max-updates 10000
```

Reduce GPU memory usage by lowering the micro-batch size while increasing
gradient accumulation to retain an effective batch of 256 sentence pairs:

```bash
uv run python notebooks/00_transformer_components/train_transformer.py \
  --micro-batch-size 32 --gradient-accumulation-steps 8
```

The most important defaults are:

- micro-batch size: 64
- gradient accumulation: 4
- validation interval: 500 optimizer updates
- checkpoint interval: 100 optimizer updates
- per-invocation time budget: 8 hours
- automatic resume: enabled

Use `Ctrl-c` once when you want to stop manually. The script finishes the
current optimizer update, saves `latest.pt`, and exits cleanly.

## Generate validation predictions

Stop the training process before evaluation so both models do not compete for
GPU memory. The evaluation script loads `best.pt` by default, reconstructs the
same 5,000-example validation split produced with seed 42, and performs batched
greedy decoding:

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

If training used a different `--seed` or `--validation-examples` value, pass
the same values to the evaluation script. Use `--checkpoint` to evaluate
another checkpoint and `--batch-size` to adjust GPU memory consumption. The
script writes to a `.tmp` file while generation is in progress and renames it
to the requested output only after all examples finish. If interrupted, the
partial `.tmp` file is retained.
