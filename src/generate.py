"""Generate ID-aligned translations and per-example inference timings."""
from __future__ import annotations

import argparse
import json
import time

import sentencepiece as spm
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from src.common import CHECKPOINTS, DATA, RESULTS, config, count_parameters, read_jsonl, set_seed, write_json, write_jsonl
from src.models.transformer import TranslationTransformer, translate


INSTRUCTION = "Translate the English text into Hindi.\nReturn only the Hindi translation.\n\nEnglish text:\n"


def load_system(name, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = RESULTS / "environment.json"
    revisions = json.loads(environment.read_text(encoding="utf-8")).get("models", {}) if environment.exists() else {}
    if name == "transformer":
        folder = CHECKPOINTS / "transformer"
        tokenizer = spm.SentencePieceProcessor(model_file=str(folder / "tokenizer.model"))
        settings = cfg["transformer"]
        model = TranslationTransformer(tokenizer.vocab_size(), **{
            key: settings[key] for key in ("d_model", "nhead", "encoder_layers",
                                             "decoder_layers", "ff_dim", "dropout")})
        model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=True))
        model.to(device).eval()
        return model, tokenizer, "local-best", count_parameters(model)
    if name == "mt5":
        folder = CHECKPOINTS / "mt5"
        training = json.loads((RESULTS / "training_mt5.json").read_text(encoding="utf-8"))
        revision = training["base_revision"]
        tokenizer = AutoTokenizer.from_pretrained(folder, use_fast=False)
        base = AutoModelForSeq2SeqLM.from_pretrained(cfg["mt5"]["model_id"], revision=revision)
        model = PeftModel.from_pretrained(base, folder).to(device).eval()
        parameters = count_parameters(model)
        # Inference freezes the adapter; report the count used during adaptation.
        parameters["trainable"] = training["parameters"]["trainable"]
        return model, tokenizer, revision, parameters
    model_id = cfg[name]["model_id"]
    revision = revisions.get(name, {}).get("revision")
    if device.type != "cuda":
        raise RuntimeError("4-bit Llama/Gemma inference requires CUDA in this experiment")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, quantization_config=quant, device_map="auto").eval()
    return model, tokenizer, revision, count_parameters(model)


@torch.inference_mode()
def generate_one(name, model, tokenizer, source, cfg, temperature):
    limit = cfg["max_new_tokens"]
    sampling = temperature is not None
    options = {"do_sample": sampling, "max_new_tokens": limit}
    if sampling:
        options["temperature"] = temperature
        options["top_k"] = 0
        options["top_p"] = 1.0
    if name == "transformer":
        return translate(model, tokenizer, source, cfg["source_max_tokens"], limit, temperature)
    if name == "mt5":
        text = "translate English to Hindi: " + source
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=cfg["source_max_tokens"]).to(next(model.parameters()).device)
        output = model.generate(**inputs, **options)
        raw = tokenizer.decode(output[0], skip_special_tokens=True)
        eos = tokenizer.eos_token_id
        return raw, bool(len(output[0]) >= limit and int(output[0][-1]) != eos)
    messages = [{"role": "user", "content": INSTRUCTION + source}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    output = model.generate(inputs, pad_token_id=tokenizer.eos_token_id, **options)
    continuation = output[0][inputs.shape[-1]:]
    raw = tokenizer.decode(continuation, skip_special_tokens=True)
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = eos_ids or [tokenizer.eos_token_id]
    return raw, bool(len(continuation) >= limit and int(continuation[-1]) not in eos_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["transformer", "mt5", "llama", "gemma"])
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    cfg = config()
    set_seed(cfg["seed"])
    rows = list(read_jsonl(DATA / f"{args.split}.jsonl"))
    if args.limit:
        rows = rows[:args.limit]
    run_id = args.run_id or f"{args.model}_{args.split}_{'greedy' if args.temperature is None else f't{args.temperature}'}"
    model, tokenizer, revision, parameters = load_system(args.model, cfg)
    # Warm-up has no score or timing entry.
    generate_one(args.model, model, tokenizer, rows[0]["source"], cfg, args.temperature)
    output = RESULTS / "predictions" / f"{run_id}.jsonl"
    predictions = list(read_jsonl(output)) if output.exists() else []
    completed = {row["id"] for row in predictions}
    if len(completed) != len(predictions) or any(row["run_id"] != run_id for row in predictions):
        raise ValueError("Existing prediction file has duplicate IDs or a different run ID")
    requested = {row["id"] for row in rows}
    if not completed <= requested:
        raise ValueError("Existing predictions are not a subset of requested IDs")
    for number, row in enumerate(rows, 1):
        if row["id"] in completed:
            continue
        if args.temperature is not None:
            torch.manual_seed(cfg["seed"] + number)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg["seed"] + number)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        raw, truncated = generate_one(args.model, model, tokenizer, row["source"], cfg, args.temperature)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        predictions.append({"id": row["id"], "model": args.model,
                            "source": row["source"], "reference": row["reference"],
                            "raw_output": raw, "prediction": raw.strip(),
                            "truncated": truncated, "inference_ms": (time.perf_counter()-started)*1000,
                            "run_id": run_id})
        if len(predictions) % 25 == 0:
            write_jsonl(output, predictions)
            print(f"{number}/{len(rows)}", flush=True)
    write_jsonl(output, predictions)
    write_json(RESULTS / "metrics" / f"{run_id}_run.json", {
        "model": args.model, "revision": revision, "parameters": parameters,
        "split": args.split, "examples": len(rows), "temperature": args.temperature,
        "sampling": args.temperature is not None, "seed": cfg["seed"],
        "max_new_tokens": cfg["max_new_tokens"], "timing": "wall clock per sentence including tokenization; warm-up excluded",
        "device": str(next(model.parameters()).device)})
    print(output)


if __name__ == "__main__":
    main()

