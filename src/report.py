"""Build a factual result summary after every required run completes."""
from __future__ import annotations

import json

from src.common import RESULTS, ROOT


MODELS = ["transformer", "mt5", "llama", "gemma"]


def main():
    items = {}
    for model in MODELS:
        run = f"{model}_test_greedy"
        score = json.loads((RESULTS / "metrics" / f"{run}.json").read_text(encoding="utf-8"))
        bench = json.loads((RESULTS / "metrics" / f"{run}_benchmark.json").read_text(encoding="utf-8"))
        items[model] = (score, bench)
    ids = {items[m][0]["examples"] for m in MODELS}
    if len(ids) != 1:
        raise ValueError("Model evaluation sizes differ")
    best = max(MODELS, key=lambda m: items[m][0]["chrF"])
    lines = ["# English to Hindi Translation Results", "",
             f"All four systems were scored on the same {ids.pop()} fixed IITB test examples.", "",
             "| Model | BLEU | chrF | Mean ms/sentence | Training seconds | Total parameters | Trainable parameters |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in MODELS:
        score, bench = items[model]
        parameters = bench["parameters"]
        if bench["training_seconds"] is None:
            train = "N/A"
        else:
            prefix = "≈" if bench.get("training_time_method", "").startswith("estimated") else ""
            train = f"{prefix}{bench['training_seconds']:.1f}"
        lines.append(f"| {model} | {score['BLEU']:.2f} | {score['chrF']:.2f} | {bench['mean_ms']:.1f} | {train} | {parameters['total']:,} | {parameters['trainable']:,} |")
    lines += ["", f"Highest measured chrF: **{best}**. This is a quality ranking on this test set, not an architecture-only effect.", "",
              "## Temperature study", "",
              "The following 100-example validation runs use one seed. Greedy decoding disables sampling; the other runs use `top_k=0` and `top_p=1`.", "",
              "| Model | Decoding | BLEU | chrF |",
              "| --- | --- | ---: | ---: |"]
    for model in MODELS:
        for label, suffix in (("Greedy", "greedy"), ("Temperature 0.3", "t0.3"),
                              ("Temperature 0.7", "t0.7")):
            path = RESULTS / "metrics" / f"{model}_validation_{suffix}.json"
            if path.exists():
                score = json.loads(path.read_text(encoding="utf-8"))
                lines.append(f"| {model} | {label} | {score['BLEU']:.2f} | {score['chrF']:.2f} |")
    lines += ["",
              "## Limitations", "",
              "The systems differ in pretraining, model size, tokenization, precision, and adaptation. The scratch model was trained on a small subset. Decoder-only systems used prompting, so their training time is not applicable.", "",
              "## Error and responsible use review", "",
              "Inspect [actual translation examples](qualitative/review_examples.md) for names, amounts, negation, unsupported additions, gender assumptions, cultural meaning, and sensitive wording. Consequential translations require human review. The separate temperature experiment is exploratory.", ""]
    mt5_training = json.loads((RESULTS / "training_mt5.json").read_text(encoding="utf-8"))
    if mt5_training.get("interrupted"):
        limitation = (f"mT5 used its validation-selected epoch-{mt5_training['best_epoch']} adapter. "
                      "Epoch 3 was interrupted by a CUDA error; its unfinished updates were not scored. "
                      "The reported adaptation time is an estimate from file timestamps.")
        index = lines.index("## Error and responsible use review")
        lines[index:index] = [limitation, ""]
    (RESULTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(RESULTS / "REPORT.md")


if __name__ == "__main__":
    main()

