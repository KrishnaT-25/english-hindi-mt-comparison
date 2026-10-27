"""Make measured comparison plots and a compact qualitative review sheet."""
from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt

from src.common import DATA, RESULTS, read_jsonl, write_json


MODELS = ["transformer", "mt5", "llama", "gemma"]


def main():
    metrics, predictions = {}, {}
    for model in MODELS:
        run = f"{model}_test_greedy"
        metrics[model] = json.loads((RESULTS / "metrics" / f"{run}.json").read_text(encoding="utf-8"))
        predictions[model] = {r["id"]: r for r in read_jsonl(RESULTS / "predictions" / f"{run}.jsonl")}
    plot_dir = RESULTS / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for key, title, filename in (("BLEU", "Corpus BLEU", "bleu.png"),
                                 ("chrF", "Corpus chrF", "chrf.png"),
                                 ("mean_inference_ms", "Mean inference milliseconds per sentence", "latency.png")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(MODELS, [metrics[m][key] for m in MODELS], color="#3465a4")
        ax.set_title(title)
        ax.set_ylabel(title)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)
    for model in ("transformer", "mt5"):
        path = RESULTS / f"training_{model}.json"
        if path.exists():
            history = json.loads(path.read_text(encoding="utf-8"))["history"]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([x["epoch"] for x in history], [x["train_loss"] for x in history], label="Train")
            ax.plot([x["epoch"] for x in history], [x["validation_loss"] for x in history], label="Validation")
            ax.set(xlabel="Epoch", ylabel="Loss", title=f"{model} training")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{model}_loss.png", dpi=180)
            plt.close(fig)
    examples = list(read_jsonl(DATA / "test.jsonl"))
    categories = {
        "numbers": lambda x: bool(re.search(r"\d", x)),
        "negation": lambda x: bool(re.search(r"\b(not|never|no|without)\b", x, re.I)),
        "long": lambda x: len(x.split()) >= 25,
        "general": lambda x: True,
    }
    selected, used = [], set()
    for category, predicate in categories.items():
        for row in examples:
            if row["id"] not in used and predicate(row["source"]):
                selected.append({"category": category, **row,
                                 "outputs": {m: predictions[m][row["id"]]["prediction"] for m in MODELS}})
                used.add(row["id"])
                if sum(x["category"] == category for x in selected) >= 5:
                    break
    write_json(RESULTS / "qualitative" / "review_examples.json", selected)
    lines = ["# Translation examples for manual review", ""]
    for row in selected:
        lines += [f"## {row['category']} {row['id']}", "", f"Source: {row['source']}",
                  "", f"Reference: {row['reference']}", ""]
        lines += [f"- {model}: {row['outputs'][model]}" for model in MODELS]
        lines.append("")
    (RESULTS / "qualitative" / "review_examples.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

