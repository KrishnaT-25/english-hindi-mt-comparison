"""Train the scratch model's tokenizer on the frozen train set only."""
from __future__ import annotations

import tempfile

import sentencepiece as spm

from src.common import CHECKPOINTS, DATA, ROOT, config, read_jsonl, write_json


def main():
    cfg = config()
    out = CHECKPOINTS / "transformer"
    out.mkdir(parents=True, exist_ok=True)
    temporary_root = ROOT / ".tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as temp:
        corpus = __import__("pathlib").Path(temp) / "corpus.txt"
        with corpus.open("w", encoding="utf-8") as f:
            for row in read_jsonl(DATA / "train.jsonl"):
                f.write(row["source"] + "\n" + row["reference"] + "\n")
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=str(out / "tokenizer"),
            vocab_size=cfg["transformer"]["vocab_size"], model_type="unigram",
            character_coverage=0.9995, pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            hard_vocab_limit=False,
        )
    write_json(out / "tokenizer_config.json", {"train_only": True,
                                                 "vocab_size": spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model")).vocab_size()})


if __name__ == "__main__":
    main()

