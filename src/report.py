"""Build a factual result summary after every required run completes."""
from __future__ import annotations

import argparse
import json

from src.common import RESULTS, ROOT


MODELS = ["transformer", "mt5", "llama", "gemma"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-count", type=int,
                        help="Use the shared fixed-prefix test subset instead of the full test run")
    args = parser.parse_args()
    items = {}
    for model in MODELS:
        run = (f"{model}_test_{args.test_count}_greedy" if args.test_count
               else f"{model}_test_greedy")
        score = json.loads((RESULTS / "metrics" / f"{run}.json").read_text(encoding="utf-8"))
        bench = json.loads((RESULTS / "metrics" / f"{run}_benchmark.json").read_text(encoding="utf-8"))
        items[model] = (score, bench)
    ids = {items[m][0]["examples"] for m in MODELS}
    if len(ids) != 1:
        raise ValueError("Model evaluation sizes differ")
    if args.test_count and ids != {args.test_count}:
        raise ValueError("Reported size differs from the requested test subset")
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
    lines += ["", "## Output diagnostics", "",
              "| Model | Empty | Truncated at 128 tokens | No Devanagari characters |",
              "| --- | ---: | ---: | ---: |"]
    for model in MODELS:
        score = items[model][0]
        lines.append(f"| {model} | {score['empty_outputs']} | {score['truncated_outputs']} | {score['non_hindi_outputs']} |")
    lines += ["", f"Highest measured chrF: **{best}**. This is a quality ranking on this test set, not an architecture-only effect.", "",
              "## Model comparison", ""]
    encoder_best = max(("transformer", "mt5"), key=lambda m: items[m][0]["chrF"])
    decoder_best = max(("llama", "gemma"), key=lambda m: items[m][0]["chrF"])
    fastest = min(MODELS, key=lambda m: items[m][1]["mean_ms"])
    lines += [
        f"The best encoder-decoder system here is {encoder_best} "
        f"(chrF {items[encoder_best][0]['chrF']:.2f}); the best prompt-only decoder system "
        f"is {decoder_best} (chrF {items[decoder_best][0]['chrF']:.2f}). "
        "This comparison reflects the actual trained checkpoints and prompts, not a controlled "
        "architecture ablation.", "",
        f"{fastest} has the lowest mean recorded inference time "
        f"({items[fastest][1]['mean_ms']:.1f} ms/sentence at batch size 1). "
        "The table also gives total and adapted parameter counts. The two prompt-only systems "
        "have no adaptation time; the trained systems used different procedures and data exposure.", "",
              "## Temperature study", "",
              "The following 100-example validation runs use one seed. Greedy decoding disables sampling; the other runs use `top_k=0` and `top_p=1`.", "",
              "| Model | Decoding | BLEU | chrF |",
              "| --- | --- | ---: | ---: |"]
    temperature_scores = {}
    temperature_rows = []
    for model in MODELS:
        for label, suffix in (("Greedy", "greedy"), ("Temperature 0.3", "t0.3"),
                              ("Temperature 0.7", "t0.7")):
            path = RESULTS / "metrics" / f"{model}_validation_{suffix}.json"
            score = json.loads(path.read_text(encoding="utf-8"))
            if score["examples"] != 100:
                raise ValueError(f"Temperature study must use 100 validation IDs: {path}")
            temperature_scores[(model, suffix)] = score
            temperature_rows.append(f"| {model} | {label} | {score['BLEU']:.2f} | {score['chrF']:.2f} |")
    lines += temperature_rows + [""]
    temperature_notes = []
    for model in MODELS:
        best_decoding = max(("greedy", "t0.3", "t0.7"),
                            key=lambda suffix: temperature_scores[(model, suffix)]["chrF"])
        delta = (temperature_scores[(model, best_decoding)]["chrF"]
                 - temperature_scores[(model, "greedy")]["chrF"])
        temperature_notes.append(
            f"{model}: {best_decoding} gives the highest validation chrF "
            f"({temperature_scores[(model, best_decoding)]['chrF']:.2f}; "
            f"change from greedy {delta:+.2f}).")
    lines += ["On this fixed validation subset, " + " ".join(temperature_notes),
              "These 100-example, one-seed results are exploratory and were not used to select test checkpoints. "
              "See [temperature output changes and examples](qualitative/temperature_examples.md).", "",
              "## Limitations", "",
              "The systems differ in pretraining, model size, tokenization, precision, and adaptation. The scratch model was trained on a small subset. Decoder-only systems used prompting, so their training time is not applicable.", "",
              "## Error and responsible use review", "",
              "See the [manual review of observed errors](qualitative/manual_review.md) and [actual translation examples](qualitative/review_examples.md) for names, numbers, negation, unsupported additions, gendered wording, cultural meaning, and sensitive text. Consequential translations require human review. The separate temperature experiment is exploratory.", ""]
    mt5_training = json.loads((RESULTS / "training_mt5.json").read_text(encoding="utf-8"))
    if mt5_training.get("interrupted"):
        limitation = (f"mT5 used its validation-selected epoch-{mt5_training['best_epoch']} adapter. "
                      "Epoch 3 was interrupted by a CUDA error; its unfinished updates were not scored. "
                      "The reported adaptation time is an estimate from file timestamps.")
        index = lines.index("## Error and responsible use review")
        lines[index:index] = [limitation, ""]
    if args.test_count:
        index = lines.index("## Error and responsible use review")
        lines[index:index] = [
            f"The primary comparison uses the first {args.test_count} IDs of the frozen "
            "seed-42 test selection. The Transformer, mT5, and Llama also have supplemental "
            "1,000-example prediction files, but those larger runs are not ranked against "
            "Gemma's smaller run.", ""]
    (RESULTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    if readme.count(start) != 1 or readme.count(end) != 1:
        raise ValueError("README result markers are missing or duplicated")
    prefix, rest = readme.split(start, 1)
    _, suffix = rest.split(end, 1)
    summary = lines[2:10] + ["", f"Highest measured chrF: **{best}**.", "",
                              f"The strongest encoder-decoder system here is {encoder_best} "
                              f"(chrF {items[encoder_best][0]['chrF']:.2f}); the strongest "
                              f"prompt-only decoder system is {decoder_best} "
                              f"(chrF {items[decoder_best][0]['chrF']:.2f}). "
                              f"{fastest} has the lowest mean batch-1 inference time. "
                              "Pretraining, size, tokenization, precision, and adaptation also differ, "
                              "so this is an observed system comparison rather than an architecture-only test.", "",
                              "### Sampling temperature analysis", "",
                              "All three decoding settings were scored on the same 100 validation IDs. "
                              "This one-seed comparison is exploratory.", "",
                              "| Model | Decoding | BLEU | chrF |",
                              "| --- | --- | ---: | ---: |"] + temperature_rows + ["",
                              " ".join(temperature_notes), "",
                              "See the [full measured report](results/REPORT.md), "
                              "[plots](results/plots/), [temperature examples](results/qualitative/temperature_examples.md), "
                              "and [manual error review](results/qualitative/manual_review.md).", "",
                              "The mT5 third epoch was interrupted by a CUDA error; the validation-selected "
                              "epoch-2 adapter was used. Its adaptation time is estimated from file timestamps."]
    if args.test_count:
        summary += ["", f"The primary comparison uses {args.test_count} shared frozen test IDs. "
                    "The 1,000-example Transformer, mT5, and Llama runs are supplementary."]
    readme_path.write_text(prefix + start + "\n" + "\n".join(summary) + "\n" + end + suffix,
                           encoding="utf-8")
    print(RESULTS / "REPORT.md")


if __name__ == "__main__":
    main()

