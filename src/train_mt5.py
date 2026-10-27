"""Fine-tune mT5 with a small, restorable LoRA adapter."""
from __future__ import annotations

import argparse
import json
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

from src.common import CHECKPOINTS, DATA, RESULTS, config, count_parameters, read_jsonl, set_seed, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = config()
    set_seed(cfg["seed"])
    settings = cfg["mt5"]
    environment = RESULTS / "environment.json"
    revision = json.loads(environment.read_text(encoding="utf-8"))["models"]["mt5"].get("revision") if environment.exists() else None
    tokenizer = AutoTokenizer.from_pretrained(settings["model_id"], revision=revision,
                                              use_fast=False)
    base = AutoModelForSeq2SeqLM.from_pretrained(settings["model_id"], revision=revision)
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=settings["lora_rank"],
        lora_alpha=settings["lora_alpha"], lora_dropout=settings["lora_dropout"],
        target_modules=["q", "v"], bias="none"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type != "cuda":
        raise RuntimeError("Full mT5 fine-tuning requires CUDA; use --smoke for a CPU check")
    model.to(device)

    def tokenize(row):
        item = tokenizer("translate English to Hindi: " + row["source"],
                         truncation=True, max_length=cfg["source_max_tokens"])
        item["labels"] = tokenizer(text_target=row["reference"], truncation=True,
                                    max_length=cfg["target_max_tokens"])["input_ids"]
        return item

    train = list(read_jsonl(DATA / "train.jsonl"))
    validation = list(read_jsonl(DATA / "validation.jsonl"))
    if args.smoke:
        train, validation = train[:16], validation[:8]
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)
    loaders = [DataLoader([tokenize(x) for x in rows], batch_size=settings["batch_size"],
                          shuffle=shuffle, collate_fn=collator)
               for rows, shuffle in ((train, True), (validation, False))]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=settings["lr"])
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    out = CHECKPOINTS / "mt5"
    out.mkdir(parents=True, exist_ok=True)
    best_loss, history, started = float("inf"), [], time.perf_counter()
    for epoch_number in range(1, (1 if args.smoke else settings["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_sum = 0.0
        for step, batch in enumerate(loaders[0], 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss / settings["accumulation_steps"]).backward()
            if step % settings["accumulation_steps"] == 0 or step == len(loaders[0]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_sum += loss.item()
        model.eval()
        validation_sum = 0.0
        with torch.inference_mode():
            for batch in loaders[1]:
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    validation_sum += model(**batch).loss.item()
        val_loss = validation_sum / len(loaders[1])
        history.append({"epoch": epoch_number, "train_loss": train_sum / len(loaders[0]),
                        "validation_loss": val_loss})
        print(history[-1], flush=True)
        if val_loss < best_loss and not args.smoke:
            best_loss = val_loss
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
    if not args.smoke:
        write_json(RESULTS / "training_mt5.json", {
            "history": history, "seconds": time.perf_counter() - started,
            "best_validation_loss": best_loss, "parameters": count_parameters(model),
            "base_model_id": settings["model_id"], "base_revision": revision,
            "device": str(device), "amp_dtype": str(amp_dtype) if use_amp else None,
            "config": settings})


if __name__ == "__main__":
    main()

