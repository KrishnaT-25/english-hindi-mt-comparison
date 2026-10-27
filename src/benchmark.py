"""Summarize recorded warmed inference timings and resource usage."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from src.common import RESULTS, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    rows = list(read_jsonl(args.predictions))
    if not rows:
        raise ValueError("No predictions to benchmark")
    run_id = rows[0]["run_id"]
    if any(row["run_id"] != run_id for row in rows):
        raise ValueError("Mixed run IDs")
    timing = sorted(row["inference_ms"] for row in rows)
    run = json.loads((RESULTS / "metrics" / f"{run_id}_run.json").read_text(encoding="utf-8"))
    result = {"run_id": run_id, "model": run["model"], "examples": len(rows),
              "mean_ms": statistics.mean(timing), "median_ms": statistics.median(timing),
              "p95_ms": timing[int(0.95 * (len(timing)-1))],
              "parameters": run["parameters"], "device": run["device"],
              "timing_policy": run["timing"]}
    training = RESULTS / f"training_{run['model']}.json"
    result["training_seconds"] = json.loads(training.read_text(encoding="utf-8"))["seconds"] if training.exists() else None
    write_json(RESULTS / "metrics" / f"{run_id}_benchmark.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

