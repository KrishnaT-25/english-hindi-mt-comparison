# English to Hindi Machine Translation Comparison

This repository implements a controlled comparison for ICT4403 Assignment 3. It uses one fixed English-to-Hindi test set to compare a Transformer trained from scratch, mT5 with a LoRA fine-tune, and prompt-only Llama and Gemma systems. Scores are generated from saved predictions, never entered by hand.

## Dataset and experiment

The data source is the [IIT Bombay English-Hindi Parallel Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi). The preparation script uses a pinned dataset commit, preserves its official split boundaries, normalizes Unicode to NFC, removes empty and repeated pairs, and prevents source overlap across the selected splits. With seed 42, cleaning produced 10,000 training pairs, all 520 official validation pairs, and 1,000 fixed official test pairs. Actual sizes and SHA-256 hashes are written to `data/manifests/dataset.json`.

Dataset attribution: Anoop Kunchukuttan, Pratik Mehta, and Pushpak Bhattacharyya, “The IIT Bombay English-Hindi Parallel Corpus,” LREC 2018. The [dataset repository's license](https://huggingface.co/datasets/cfilt/iitb-english-hindi/blob/main/LICENSE.md) is Creative Commons Attribution-NonCommercial 4.0. This experiment uses a cleaned, fixed subset for a noncommercial academic assignment; published examples retain the source link and attribution.

The scratch model uses a joint 8,000-token SentencePiece tokenizer trained only on the selected training data. mT5, Llama, and Gemma each use their checkpoint tokenizer. Sources and references are kept in UTF-8 JSONL with stable IDs. Checkpoints, downloaded data, and credentials are intentionally excluded from Git.

## Systems

| System | Checkpoint or architecture | Adaptation |
| --- | --- | --- |
| Basic Transformer | Four encoder and four decoder layers, width 256, four heads | Train from scratch, up to 10 epochs |
| mT5 | [`google/mt5-small`](https://huggingface.co/google/mt5-small) | LoRA fine-tune, up to 3 epochs |
| Llama | [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | Prompt-only, 4-bit inference |
| Gemma | [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) | Prompt-only, 4-bit inference |

Llama requires approved Hugging Face access. Gemma requires acceptance of its usage conditions. Use `HF_TOKEN` in the environment or `hf auth login`; do not put tokens in files or commits.

## Setup on Windows with an NVIDIA GPU

Use Python 3.13 or a supported Python 3.11/3.12 installation. The commands below assume PowerShell from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m src.environment
.\.venv\Scripts\python.exe -m src.check_access
.\.venv\Scripts\Activate.ps1
```

Install the PyTorch CUDA wheel that matches the local driver if CUDA 12.8 is unsuitable. `src.environment` saves Python, package, GPU, and model revision details. If PowerShell blocks activation, replace `python` in later commands with `.\.venv\Scripts\python.exe`. Run each model's one-sentence load/generation trial before the main test set, especially on a 4 GB GPU. The commands require network access for first downloads.

## Reproduce the experiment

```powershell
python -m src.prepare_data
python -m src.data_report
python -m src.train_tokenizer
python -m src.token_lengths
python -m unittest discover -s tests -v
python -m src.train_transformer --smoke
python -m src.train_transformer
python -m src.train_mt5 --smoke
python -m src.train_mt5
python -m src.generate transformer
python -m src.generate mt5
python -m src.generate llama
python -m src.generate gemma
```

The smoke runs must finish before the long runs. Train and generate one model at a time to fit this laptop. Each generator saves its complete predictions and per-sentence wall time, including tokenization, after a warm-up. The main comparison uses greedy decoding with a 128-new-token limit. Read `truncated`, `empty_outputs`, and `non_hindi_outputs` alongside BLEU and chrF.

```powershell
foreach ($m in 'transformer','mt5','llama','gemma') {
  python -m src.evaluate "results/predictions/${m}_test_greedy.jsonl"
}
python -m src.analyze
foreach ($m in 'transformer','mt5','llama','gemma') {
  python -m src.benchmark "results/predictions/${m}_test_greedy.jsonl"
}
python -m src.report
```

For an exploratory temperature study, use the first 100 fixed validation IDs for each model. The sampled runs use seed 42, temperatures 0.3 or 0.7, `top_k=0`, and `top_p=1`. Greedy runs are the deterministic baseline.

```powershell
foreach ($m in 'transformer','mt5','llama','gemma') {
  python -m src.generate $m --split validation --limit 100
  python -m src.generate $m --split validation --limit 100 --temperature 0.3
  python -m src.generate $m --split validation --limit 100 --temperature 0.7
  foreach ($run in 'greedy','t0.3','t0.7') {
    python -m src.evaluate "results/predictions/${m}_validation_${run}.jsonl" --split validation
  }
}
```

## Results and interpretation

The scratch Transformer test run is complete. Its corpus BLEU is **0.29** and chrF is **10.23** on 1,000 fixed test examples; 130 outputs hit the token limit and 53 lacked Devanagari characters. This is poor translation quality. The mT5, Llama, and Gemma comparisons are still running or pending access, so no overall winner is claimed yet. See [`transformer_review.md`](results/qualitative/transformer_review.md) for observed errors.

The following files are the authoritative outputs after a successful run:

- `results/predictions/*_test_greedy.jsonl`: full raw and scored translations for the identical test IDs.
- `results/metrics/*_test_greedy.json`: corpus BLEU, chrF, missing-Hindi and truncation counts, and mean inference milliseconds.
- `results/metrics/*_run.json`: model revisions, parameters, precision-related configuration, and measurement conditions.
- `results/training_*.json`: training duration, curves, and checkpoint selection.
- `results/plots/*.png` and `results/qualitative/review_examples.md`: measured plots and examples for manual review.

The best model must be identified from the completed measurements. Performance differences cannot be attributed to architecture alone because pretraining, parameter counts, tokenization, precision, and adaptation differ. Prompt-only systems have no training time in this experiment. A 10,000-pair scratch run is resource-limited and may translate poorly.

Manual review should check names, numbers, negation, added details, gender assumptions, cultural expressions, and sensitive content using actual predictions. Translation intended for consequential use needs human review, terminology checks, and evaluation across relevant groups. The temperature comparison is exploratory because it uses one seed and a small validation sample.

## Submission

Publish this repository publicly with code, manifests, result files, plots, and examples. Do not commit downloaded model weights or credentials. The separate cover sheet contains the student details, dataset URL, and GitHub repository URL, and its PDF is the LMS upload artifact.

