"""Make measured comparison plots and a compact qualitative review sheet."""
from __future__ import annotations

import argparse
import json
import re

import matplotlib.pyplot as plt

from src.common import DATA, RESULTS, read_jsonl, write_json


MODELS = ["transformer", "mt5", "llama", "gemma"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-count", type=int)
    args = parser.parse_args()
    metrics, predictions = {}, {}
    for model in MODELS:
        run = (f"{model}_test_{args.test_count}_greedy" if args.test_count
               else f"{model}_test_greedy")
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
    if args.test_count:
        examples = examples[:args.test_count]
    categories = {
        "names": lambda x: bool(re.search(r"\b[A-Z][a-z]{2,}\b", x[1:])),
        "numbers": lambda x: bool(re.search(r"\d", x)),
        "negation": lambda x: bool(re.search(r"\b(not|never|no|without)\b", x, re.I)),
        "gendered wording": lambda x: bool(re.search(
            r"\b(she|he|her|his|woman|women|man|men|wife|husband|girl|boy)\b", x, re.I)),
        "cultural references": lambda x: bool(re.search(
            r"\b(temple|mosque|church|festival|prayer|god|goddess|wedding|ritual|tradition)\b", x, re.I)),
        "sensitive content": lambda x: bool(re.search(
            r"\b(death|died|killed|murder|prison|sexual|abuse|violence|religion|caste)\b", x, re.I)),
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
    lines = ["# Translation examples for manual review", "",
             "Categories are source-text search cues, not automatic error or bias findings. "
             "Compare each output with the source and reference before drawing conclusions.", ""]
    for row in selected:
        lines += [f"## {row['category']} {row['id']}", "", f"Source: {row['source']}",
                  "", f"Reference: {row['reference']}", ""]
        lines += [f"- {model}: {row['outputs'][model]}" for model in MODELS]
        lines.append("")
    (RESULTS / "qualitative" / "review_examples.md").write_text("\n".join(lines), encoding="utf-8")

    validation = list(read_jsonl(DATA / "validation.jsonl"))[:100]
    validation_ids = [row["id"] for row in validation]
    temperature_lines = ["# Sampling temperature examples", "",
                         "All runs use the same first 100 frozen validation IDs and seed 42. "
                         "Changed-output counts compare exact scored strings with greedy decoding; "
                         "they are not quality measures.", ""]
    temperature_chrf = {}
    for model in MODELS:
        runs = {}
        for suffix in ("greedy", "t0.3", "t0.7"):
            rows = list(read_jsonl(RESULTS / "predictions" /
                                   f"{model}_validation_{suffix}.jsonl"))
            runs[suffix] = {row["id"]: row for row in rows}
            if len(runs[suffix]) != len(rows) or set(runs[suffix]) != set(validation_ids):
                raise ValueError(f"Temperature ID mismatch: {model} {suffix}")
        temperature_lines += [f"## {model}", ""]
        temperature_chrf[model] = [json.loads((RESULTS / "metrics" /
            f"{model}_validation_{suffix}.json").read_text(encoding="utf-8"))["chrF"]
            for suffix in ("greedy", "t0.3", "t0.7")]
        for suffix in ("t0.3", "t0.7"):
            changed = sum(runs[suffix][id_]["prediction"] !=
                          runs["greedy"][id_]["prediction"] for id_ in validation_ids)
            temperature_lines.append(f"- {suffix}: {changed}/100 outputs differ from greedy")
        temperature_lines.append("")
        selected = [id_ for id_ in validation_ids if any(
            runs[suffix][id_]["prediction"] != runs["greedy"][id_]["prediction"]
            for suffix in ("t0.3", "t0.7"))][:2]
        for id_ in selected:
            row = runs["greedy"][id_]
            temperature_lines += [f"### {id_}", "", f"Source: {row['source']}", "",
                                  f"Reference: {row['reference']}", ""]
            for suffix in ("greedy", "t0.3", "t0.7"):
                prediction = runs[suffix][id_]["prediction"].replace("\n", " ")
                temperature_lines.append(f"- {suffix}: {prediction}")
            temperature_lines.append("")
    (RESULTS / "qualitative" / "temperature_examples.md").write_text(
        "\n".join(temperature_lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in MODELS:
        ax.plot((0, 1, 2), temperature_chrf[model], marker="o", label=model)
    ax.set_xticks((0, 1, 2), ("Greedy", "0.3", "0.7"))
    ax.set(xlabel="Decoding setting", ylabel="Validation chrF",
           title="Sampling temperature study (100 fixed validation IDs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "temperature_chrf.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

