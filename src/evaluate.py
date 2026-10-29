"""Score one complete, ID-aligned prediction file with corpus BLEU and chrF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sacrebleu

from src.common import DATA, RESULTS, ROOT, read_jsonl, write_json


def expected_ids(manifest, split, subset_size, row_count):
    if split != "test" and subset_size is not None:
        raise ValueError("--subset-size is only for the test split")
    if subset_size is not None:
        if not 1 <= subset_size <= len(manifest["ids"]):
            raise ValueError("Invalid test subset size")
        if row_count != subset_size:
            raise ValueError("Prediction row count differs from the declared test subset size")
        return manifest["ids"][:subset_size]
    return manifest["ids"] if split == "test" else manifest["ids"][:row_count]


def score(rows, references):
    by_id = {}
    for row in rows:
        if row["id"] in by_id:
            raise ValueError(f"Duplicate prediction ID: {row['id']}")
        by_id[row["id"]] = row
    expected = {row["id"]: row for row in references}
    if set(by_id) != set(expected):
        raise ValueError(f"ID mismatch: missing={len(set(expected)-set(by_id))}, extra={len(set(by_id)-set(expected))}")
    ids = sorted(expected)
    predictions = [by_id[id_]["prediction"] for id_ in ids]
    targets = [expected[id_]["reference"] for id_ in ids]
    for id_ in ids:
        if by_id[id_].get("reference") != expected[id_]["reference"]:
            raise ValueError(f"Reference mismatch: {id_}")
    bleu_metric = sacrebleu.metrics.BLEU()
    chrf_metric = sacrebleu.metrics.CHRF()
    bleu = bleu_metric.corpus_score(predictions, [targets])
    chrf = chrf_metric.corpus_score(predictions, [targets])
    return {"examples": len(ids), "BLEU": bleu.score, "chrF": chrf.score,
            "bleu_signature": str(bleu_metric.get_signature()),
            "chrf_signature": str(chrf_metric.get_signature()),
            "empty_outputs": sum(not x.strip() for x in predictions),
            "truncated_outputs": sum(bool(by_id[id_].get("truncated")) for id_ in ids),
            "non_hindi_outputs": sum(not any("\u0900" <= c <= "\u097f" for c in x) for x in predictions),
            "mean_inference_ms": sum(by_id[id_].get("inference_ms", 0) for id_ in ids) / len(ids),
            "model": by_id[ids[0]].get("model"), "run_id": by_id[ids[0]].get("run_id")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--subset-size", type=int,
                        help="Explicit fixed-prefix test subset size; required for partial test files")
    args = parser.parse_args()
    rows = list(read_jsonl(args.predictions))
    manifest = json.loads((ROOT / "data" / "manifests" / f"{args.split}_ids.json").read_text(encoding="utf-8"))
    selected_ids = expected_ids(manifest, args.split, args.subset_size, len(rows))
    if {row["id"] for row in rows} != set(selected_ids):
        raise ValueError("Prediction IDs do not match the frozen split selection")
    for row in rows:
        digest = hashlib.sha256(row["reference"].encode("utf-8")).hexdigest()
        if digest != manifest["reference_sha256"][row["id"]]:
            raise ValueError(f"Reference hash mismatch: {row['id']}")
    if (DATA / f"{args.split}.jsonl").exists():
        references = list(read_jsonl(DATA / f"{args.split}.jsonl"))
        if args.split == "validation" or args.subset_size is not None:
            references = references[:len(rows)]
    else:
        references = [{"id": row["id"], "reference": row["reference"]} for row in rows]
    result = score(rows, references)
    output = RESULTS / "metrics" / f"{args.predictions.stem}.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

