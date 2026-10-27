"""Summarize frozen data and provide a fixed alignment-inspection sample."""
from __future__ import annotations

import json
import random
import statistics

from src.common import DATA, RESULTS, ROOT, config, read_jsonl, write_json


def lengths(rows, key):
    counts = sorted(len(row[key].split()) for row in rows)
    return {"min": counts[0], "median": statistics.median(counts),
            "p95": counts[int(0.95 * (len(counts)-1))], "max": counts[-1]}


def main():
    manifest = json.loads((ROOT / "data" / "manifests" / "dataset.json").read_text(encoding="utf-8"))
    rows = {name: list(read_jsonl(DATA / f"{name}.jsonl")) for name in ("train", "validation", "test")}
    result = {"splits": manifest["splits"],
              "word_lengths": {name: {key: lengths(items, key) for key in ("source", "reference")}
                               for name, items in rows.items()}}
    rng = random.Random(config()["seed"])
    result["alignment_sample"] = rng.sample(rows["train"], min(25, len(rows["train"])))
    write_json(RESULTS / "data_analysis.json", result)
    lines = ["# Frozen data analysis", "",
             f"Dataset revision: `{manifest['dataset_revision']}`. Selection seed: {manifest['seed']}.", ""]
    for name in ("train", "validation", "test"):
        stats = manifest["splits"][name]
        lines += [f"## {name.title()}", "", f"Selected: {stats['selected']}; scanned: {stats['scanned']}.",
                  f"Removed while scanning: {stats['removed']}.",
                  f"Source word lengths: {result['word_lengths'][name]['source']}.",
                  f"Reference word lengths: {result['word_lengths'][name]['reference']}.", ""]
    lines += ["## Alignment sample", ""]
    for row in result["alignment_sample"]:
        lines += [f"- {row['id']}: {row['source']} → {row['reference']}"]
    (RESULTS / "data_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

