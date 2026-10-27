"""Freeze a leak-free subset of the official IITB splits."""
from __future__ import annotations

import argparse
import hashlib
import platform
import unicodedata

from datasets import load_dataset
from huggingface_hub import HfApi

from src.common import DATA, ROOT, config, sha256, write_json, write_jsonl


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def pair(row):
    translation = row["translation"]
    return normalize(translation["en"]), normalize(translation["hi"])


def collect(split, name, forbidden_sources=None, limit=None):
    forbidden_sources = forbidden_sources or set()
    seen_pairs, seen_sources, rows = set(), set(), []
    removed = {"empty": 0, "duplicate_pair": 0, "source_overlap": 0}
    scanned = 0
    for index, item in enumerate(split):
        scanned += 1
        source, reference = pair(item)
        if not source or not reference:
            removed["empty"] += 1
            continue
        if source in forbidden_sources:
            removed["source_overlap"] += 1
            continue
        if (source, reference) in seen_pairs or source in seen_sources:
            removed["duplicate_pair"] += 1
            continue
        seen_pairs.add((source, reference))
        seen_sources.add(source)
        rows.append({"id": f"{name}_{index:07d}", "source": source,
                     "reference": reference})
        if limit is not None and len(rows) >= limit:
            break
    return rows, {"scanned": scanned, "selected": len(rows), "removed": removed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--test-size", type=int)
    args = parser.parse_args()
    cfg = config()
    seed = cfg["seed"]
    train_size = args.train_size or cfg["train_size"]
    test_size = args.test_size or cfg["test_size"]
    repo = cfg["dataset"]
    revision = HfApi().dataset_info(repo, revision=cfg["dataset_revision"]).sha
    dataset = load_dataset(repo, revision=revision, cache_dir=str(ROOT / "data" / "cache"))

    # Preserve official split boundaries. Validation has priority when a source
    # appears in both held-out splits; train is checked against both full splits.
    validation, val_stats = collect(dataset["validation"], "validation")
    val_sources = {row["source"] for row in validation}
    test, test_stats = collect(dataset["test"].shuffle(seed=seed), "test",
                               val_sources, test_size)
    held_out = val_sources | {row["source"] for row in test}
    train, train_stats = collect(dataset["train"].shuffle(seed=seed), "train",
                                 held_out, train_size)
    if len(train) != train_size or len(test) != test_size:
        raise RuntimeError("The requested sample size is unavailable after cleaning")
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        write_jsonl(DATA / f"{name}.jsonl", rows)
        if name != "train":
            write_json(ROOT / "data" / "manifests" / f"{name}_ids.json", {
                "ids": [row["id"] for row in rows],
                "reference_sha256": {row["id"]: hashlib.sha256(row["reference"].encode("utf-8")).hexdigest()
                                     for row in rows}})
    manifest = {
        "dataset": repo, "dataset_revision": revision, "seed": seed,
        "official_split_sizes": {name: len(dataset[name]) for name in dataset},
        "rules": ["Unicode NFC", "strip surrounding whitespace", "remove empty pairs",
                  "one reference per distinct source per split", "remove source overlap",
                  "shuffle official train/test with fixed seed before selection"],
        "splits": {"train": train_stats, "validation": val_stats, "test": test_stats},
        "sha256": {name: sha256(DATA / f"{name}.jsonl")
                   for name in ("train", "validation", "test")},
        "python": platform.python_version(),
    }
    write_json(ROOT / "data" / "manifests" / "dataset.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()

