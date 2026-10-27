"""Measure selected data lengths under both training tokenizers."""
from __future__ import annotations

import json

import sentencepiece as spm
from transformers import AutoTokenizer

from src.common import CHECKPOINTS, DATA, RESULTS, config, read_jsonl, write_json


def summary(lengths, limit):
    ordered = sorted(lengths)
    return {"median": ordered[len(ordered)//2], "p95": ordered[int(0.95*(len(ordered)-1))],
            "max": ordered[-1], "over_limit": sum(x > limit for x in ordered),
            "examples": len(ordered)}


def main():
    cfg = config()
    environment = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))
    revision = environment["models"]["mt5"]["revision"]
    scratch = spm.SentencePieceProcessor(model_file=str(CHECKPOINTS / "transformer" / "tokenizer.model"))
    mt5 = AutoTokenizer.from_pretrained(cfg["mt5"]["model_id"], revision=revision,
                                         use_fast=False)
    result = {}
    for split in ("train", "validation", "test"):
        rows = list(read_jsonl(DATA / f"{split}.jsonl"))
        result[split] = {}
        for field, limit in (("source", cfg["source_max_tokens"]),
                             ("reference", cfg["target_max_tokens"])):
            scratch_lengths = [len(scratch.encode(r[field])) + 2 for r in rows]
            mt5_lengths = [len(mt5(r[field])["input_ids"]) for r in rows]
            result[split][field] = {"scratch": summary(scratch_lengths, limit),
                                    "mt5": summary(mt5_lengths, limit)}
    write_json(RESULTS / "token_lengths.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

