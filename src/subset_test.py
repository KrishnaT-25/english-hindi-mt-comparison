"""Freeze the same fixed-prefix test subset from saved full-run predictions."""
from __future__ import annotations

import argparse
import json

from src.common import RESULTS, ROOT, read_jsonl, write_json, write_jsonl


MODELS = ("transformer", "mt5", "llama", "gemma")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data" / "manifests" / "test_ids.json").read_text(encoding="utf-8"))
    if not 1 <= args.count <= len(manifest["ids"]):
        raise ValueError("Subset count is outside the frozen test selection")
    ids = manifest["ids"][:args.count]
    write_json(ROOT / "data" / "manifests" / f"test_{args.count}_ids.json", {
        "selection": f"first {args.count} IDs of the seed-42 frozen test selection",
        "ids": ids,
        "reference_sha256": {id_: manifest["reference_sha256"][id_] for id_ in ids},
    })

    for model in MODELS:
        parent_run = f"{model}_test_greedy"
        run_id = f"{model}_test_{args.count}_greedy"
        source = RESULTS / "predictions" / f"{parent_run}.jsonl"
        rows = list(read_jsonl(source))
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows) or not set(ids) <= set(by_id):
            raise ValueError(f"{model} has duplicate or missing test IDs")
        subset = [{**by_id[id_], "run_id": run_id,
                   "parent_run_id": parent_run} for id_ in ids]
        write_jsonl(RESULTS / "predictions" / f"{run_id}.jsonl", subset)

        metadata = json.loads((RESULTS / "metrics" / f"{parent_run}_run.json").read_text(encoding="utf-8"))
        metadata["examples"] = args.count
        metadata["run_id"] = run_id
        metadata["parent_run_id"] = parent_run
        metadata["selection"] = f"first {args.count} frozen test IDs"
        metadata["batch_size"] = 1
        if model in ("llama", "gemma"):
            metadata["parameters"]["trainable"] = 0
            metadata["adaptation"] = "prompt-only"
            metadata["quantization"] = "bitsandbytes NF4 4-bit, float16 compute"
        write_json(RESULTS / "metrics" / f"{run_id}_run.json", metadata)
        print(f"{model}: {len(subset)} fixed test predictions", flush=True)


if __name__ == "__main__":
    main()
