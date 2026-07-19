# Transformer component notebooks

These notebooks build an English-to-German Transformer from its individual
components. Use `4_transformer.ipynb` interactively for the educational model
walkthrough and smoke test. Long runs belong in `train_transformer.py`.

The workflow below uses the WMT14 English-German recipe, a boundary-aware tokenizer, token batches, five retained checkpoints, checkpoint averaging, beam search, and standard translation metrics.

## 1. Prepare WMT14 English-German

The preparation script downloads the same three training sources used by the
original Transformer recipe:

- Europarl v7
- Common Crawl
- News Commentary v9

It uses newstest2013 as validation and newstest2014 as the final test set. It
also removes empty, extreme-length, badly imbalanced, exact-duplicate, and
validation/test-identical pairs.

The official archives total about 1.7 GB. Allow several additional gigabytes
for extraction and prepared files:

```bash
uv run python notebooks/00_transformer_components/prepare_wmt14_en_de.py
```

Prepared files are written under:

```text
data/processed/wmt14_en_de/train.en
data/processed/wmt14_en_de/train.de
data/processed/wmt14_en_de/validation.en
data/processed/wmt14_en_de/validation.de
data/processed/wmt14_en_de/test.en
data/processed/wmt14_en_de/test.de
data/processed/wmt14_en_de/metadata.json
```

Use `--max-training-pairs 10000` only for a preparation smoke test. Do not
train the final model on that limited output.

## 2. Train the boundary-aware tokenizer

Train a joint 37,000-token English-German byte-level BPE vocabulary:

```bash
uv run python notebooks/00_transformer_components/train_wmt_tokenizer.py
```

This tokenizer preserves word boundaries and round-trips unseen Unicode bytes. 
The script also writes `train.lengths.pt`, a compact cache used to form length-bucketed token
batches without loading millions of sentences into Python objects.

Outputs:

```text
data/tokenizers/wmt14_bpe_en_de.json
data/processed/wmt14_en_de/train.lengths.pt
```

Changing the tokenizer or context length requires rebuilding the length cache.
Training rejects a cache created by another tokenizer.

## 3. Train

Create a persistent terminal session:

```bash
tmux new -s transformer
```

Start a clean run:

```bash
mkdir -p checkpoints/transformer_en_de
uv run python notebooks/00_transformer_components/train_transformer.py \
  --no-resume \
  2>&1 | tee -a checkpoints/transformer_en_de/training.log
```

The main defaults are:

- up to 4,096 padded source-plus-target tokens per micro-batch
- 8-step gradient accumulation, approximately 32,768 padded tokens per update
- maximum 256 sentence pairs per micro-batch
- 256-token source and target context
- 4,000 warmup updates
- validation every 1,000 updates on newstest2013
- retain the five checkpoints with the lowest validation losses
- early stopping after 10 validation checks without a 0.001 improvement
- maximum 20 epochs and eight hours per invocation

The token limit is a memory ceiling rather than an exact token count. If the
GPU runs out of memory, lower `--micro-batch-tokens` and increase
`--gradient-accumulation-steps` proportionally.

Training writes:

```text
checkpoints/transformer_en_de/latest.pt
checkpoints/transformer_en_de/best.pt
checkpoints/transformer_en_de/best_update_*_loss_*.pt
checkpoints/transformer_en_de/history.jsonl
```

`latest.pt` contains optimizer and resume state. The ranked
`best_update_*.pt` files contain model weights and evaluation metadata.
Starting the same command again without `--no-resume` resumes safely.

Use `Ctrl-c` once to request a checkpoint after the current optimizer update.
Detach from tmux with `Ctrl-b`, then `d`, and reconnect with:

```bash
tmux attach -t transformer
```

## 4. Average the five best checkpoints

After training stops:

```bash
uv run python \
  notebooks/00_transformer_components/average_transformer_checkpoints.py
```

This reads the ranked manifest from `latest.pt`, averages floating-point
model parameters from the five lowest-loss checkpoints, validates non-floating
buffers, and writes:

```text
checkpoints/transformer_en_de/averaged.pt
```

## 5. Evaluate with beam search and translation metrics

Evaluate the averaged model on the untouched newstest2014 test set:

```bash
uv run python notebooks/00_transformer_components/evaluate_transformer.py
```

Defaults match the original Transformer decoding recipe: beam size 4 and
length penalty 0.6. Evaluation writes both sentence-level JSONL predictions
and a metrics summary:

```text
evaluations/transformer_en_de/newstest2014_beam4.jsonl
evaluations/transformer_en_de/newstest2014_beam4.metrics.json
```

The metrics file records sacreBLEU, chrF++, reproducibility signatures,
checkpoint, split, beam size, and length penalty. To evaluate newstest2013 or
compare greedy decoding:

```bash
uv run python notebooks/00_transformer_components/evaluate_transformer.py \
  --split validation \
  --output evaluations/transformer_en_de/newstest2013_beam4.jsonl

uv run python notebooks/00_transformer_components/evaluate_transformer.py \
  --beam-size 1 \
  --length-penalty 0 \
  --output evaluations/transformer_en_de/newstest2014_greedy.jsonl
```

Use `--max-examples 20` for an end-to-end smoke test.

## 6. Translate interactively

Load the averaged model once, then enter as many English sentences as you
like. Press `Ctrl-D` or `Ctrl-C` to exit:

```bash
uv run python notebooks/00_transformer_components/translate_interactive.py
```

You can also translate one sentence non-interactively:

```bash
uv run python notebooks/00_transformer_components/translate_interactive.py \
  "How are you?"
```

## 7. Optional Hugging Face publishing

Open `5_publish_transformer_to_hf.ipynb` only after reviewing the averaged
model, test metrics, and translations. It exports a self-contained bundle and
locally reloads it before any upload.

`publish_to_hub = False` is the default. Authenticate outside the notebook
with `hf auth login` or `HF_TOKEN`, and enable the flag only when the model
is ready to publish.
