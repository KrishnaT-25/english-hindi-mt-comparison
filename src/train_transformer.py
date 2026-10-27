"""Train and validate the scratch Transformer."""
from __future__ import annotations

import argparse
import time

import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.common import CHECKPOINTS, DATA, RESULTS, config, count_parameters, read_jsonl, set_seed, write_json
from src.models.transformer import TranslationTransformer, encode_text


def make_loader(rows, tokenizer, batch_size, source_limit, target_limit, shuffle):
    def collate(batch):
        source = [encode_text(tokenizer, row["source"], source_limit, True) for row in batch]
        target = [encode_text(tokenizer, row["reference"], target_limit, False) for row in batch]
        pad = lambda seqs: nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in seqs],
            batch_first=True, padding_value=0)
        target = pad(target)
        return pad(source), target[:, :-1], target[:, 1:]
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def epoch(model, loader, optimizer, device, accumulation):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss, steps = 0.0, 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, (source, target_input, labels) in enumerate(loader, 1):
        source, target_input, labels = source.to(device), target_input.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(source, target_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            if training:
                (loss / accumulation).backward()
        if training and (step % accumulation == 0 or step == len(loader)):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = config()
    set_seed(cfg["seed"])
    settings = cfg["transformer"]
    out = CHECKPOINTS / "transformer"
    tokenizer = spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model"))
    model = TranslationTransformer(tokenizer.vocab_size(), **{
        key: settings[key] for key in ("d_model", "nhead", "encoder_layers",
                                         "decoder_layers", "ff_dim", "dropout")})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.smoke and device.type != "cuda":
        raise RuntimeError("Full Transformer training requires CUDA; use --smoke for a CPU check")
    model.to(device)
    train = list(read_jsonl(DATA / "train.jsonl"))
    validation = list(read_jsonl(DATA / "validation.jsonl"))
    if args.smoke:
        train, validation = train[:20], train[:20]
    make = lambda rows, shuffle: make_loader(rows, tokenizer, settings["batch_size"],
                                             cfg["source_max_tokens"], cfg["target_max_tokens"], shuffle)
    train_loader, validation_loader = make(train, True), make(validation, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
    best_loss, stale, history = float("inf"), 0, []
    started = time.perf_counter()
    for number in range(1, (10 if args.smoke else settings["epochs"]) + 1):
        train_loss = epoch(model, train_loader, optimizer, device, settings["accumulation_steps"])
        validation_loss = epoch(model, validation_loader, None, device, 1)
        history.append({"epoch": number, "train_loss": train_loss,
                        "validation_loss": validation_loss})
        print(history[-1], flush=True)
        if validation_loss < best_loss:
            best_loss, stale = validation_loss, 0
            if not args.smoke:
                torch.save(model.state_dict(), out / "best.pt")
        else:
            stale += 1
        if not args.smoke and stale >= settings["patience"]:
            break
    if args.smoke and history[-1]["train_loss"] >= history[0]["train_loss"]:
        raise RuntimeError("Scratch-model smoke run did not reduce training loss")
    if not args.smoke:
        write_json(RESULTS / "training_transformer.json", {
            "history": history, "seconds": time.perf_counter() - started,
            "best_validation_loss": best_loss, "parameters": count_parameters(model),
            "device": str(device), "config": settings})


if __name__ == "__main__":
    main()

