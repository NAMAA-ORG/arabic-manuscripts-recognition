# pip uninstall -y torchao   # Kaggle's preinstalled torchao is too old for peft
# pip install -U torch transformers accelerate peft bitsandbytes pillow pandas jiwer

"""QLoRA fine-tuning of a vision-language model (default: Gemma 4 E4B) for manuscript OCR.

Built for Kaggle (2x T4). Training runs on GPU 0; evaluation uses inference.run()
on both GPUs, so validation CER is directly comparable with the benchmark.

Trains on the NAMAA pages of this repository only (dataset_relative.csv) and scores
the adapters on held-out NAMAA pages.

Validation split (--split):
    page  --val_per_doc pages from every group (default; same hands/books in train and val)
    doc   whole groups held out (harder: tests transfer to unseen hands/books)

Every epoch's adapter is scored by CER on the validation pages, and the epoch with the
lowest corpus CER is copied to lora_namaa_best (validation loss did not predict CER).

Usage:
    python finetune_namaa.py --csv dataset_relative.csv --root . -o lora_out
    python finetune_namaa.py --csv dataset_relative.csv --root . --epochs 2 --no_baseline

Outputs (in --output):
    lora_namaa_ep<N>/, lora_namaa_best/            LoRA adapters (peft)
    results_<preset>-val-base.csv                  baseline CER on the validation pages
    results_<preset>-val-lora-namaa-ep<N>.csv    CER of each epoch's adapter
    epoch_scores.csv, train_history.csv, split.csv, val.csv

Use an adapter afterwards:
    python inference.py --preset gemma4-e4b --quant 4bit --device_map per-gpu --adapter lora_out/lora_namaa_best
"""

import argparse
import gc
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less memory fragmentation

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference as K  # noqa: E402


# LoRA is applied to these projections of the language model only (vision / audio stay frozen)
LORA_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


# ================= DATA =================

def arabic_fraction(text) -> float:
    """Share of non-space characters in the Arabic Unicode block."""
    chars = re.sub(r"\s", "", str(text))
    return sum("\u0600" <= c <= "\u06ff" for c in chars) / max(len(chars), 1)


def load_namaa(csv_path: Path, root: Path, split: str, val_per_doc: int, val_frac: float, seed: int):
    """Read dataset_relative.csv and split it into train / validation pages.

    Groups are the folders under Files_demo/ (one annotator's hand / book each).
    split="page": val_per_doc pages from every group; split="doc": whole groups held out.
    """
    df = pd.read_csv(csv_path)
    df["rel"] = df.path.str.replace("\\", "/", regex=False)
    df["doc"] = df.rel.map(lambda p: p.split("/")[1] if p.count("/") >= 2 else "all")
    df["img"] = df.rel.map(lambda p: str(root / p))
    df["text"] = df.main_text.astype(str)

    bad = df.text.map(arabic_fraction) < 0.6
    print(f"Dropping garbled/empty references: {int(bad.sum())}")
    df = df[~bad].reset_index(drop=True)

    if split == "page":
        # Same groups in train and val, different pages; at least one page per group stays in train
        val_idx = [
            i for _, g in df.groupby("doc")
            for i in g.sample(min(val_per_doc, len(g) - 1), random_state=seed).index
        ]
        val_df = df.loc[val_idx]
        train_df = df.drop(val_df.index)
    else:
        docs = sorted(df.doc.unique())
        random.Random(seed).shuffle(docs)

        val_docs, n = [], 0
        for doc in docs[:-1]:  # always keep at least one group for training
            if n >= val_frac * len(df):
                break
            val_docs.append(doc)
            n += (df.doc == doc).sum()

        val_df = df[df.doc.isin(val_docs)]
        train_df = df[~df.doc.isin(val_docs)]

    counts = pd.DataFrame({"train": train_df.doc.value_counts(), "val": val_df.doc.value_counts()})
    print(counts.fillna(0).astype(int).to_string())
    print(f"split={split}  NAMAA train={len(train_df)}  val={len(val_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ================= TRAINING =================

def chat(processor, prompt, image, target=None):
    """Chat-template text for one page; with `target`, the assistant turn is included."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=target is None, enable_thinking=False)


def make_builder(ocr):
    """Returns build(image_path, target) -> model inputs with labels on the transcription only."""
    processor, cfg = ocr.processor, ocr.cfg

    # Whatever the template appends after the assistant text (end-of-turn marker)
    probe = chat(processor, cfg.prompt, Image.new("RGB", (64, 64), "white"), "XYZXYZ")
    end = probe.split("XYZXYZ")[-1]

    def build(image_path, target):
        image = K.cap_pixels(Image.open(image_path).convert("RGB"), cfg.max_pixels)
        prompt = chat(processor, cfg.prompt, image)
        full = chat(processor, cfg.prompt, image, target)
        if not full.startswith(prompt):  # e.g. the template inserts/strips thinking tags in the history
            full = prompt + target + end

        enc = processor(text=[full], images=[image], return_tensors="pt")
        prompt_len = processor(text=[prompt], images=[image], return_tensors="pt")["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[:, :prompt_len] = -100  # loss only on the transcription + end-of-turn
        enc["labels"] = labels
        return enc

    return build, end


def eval_overrides(args):
    """Model settings shared by training and evaluation."""
    return dict(quant=args.quant, device_map="per-gpu", image_tokens=args.image_tokens, dtype=args.dtype)


def train_run(name, frame, val_df, args):
    """Train one LoRA adapter on `frame` (img, text) and save lora_<name>_ep<N> after every epoch.
    Returns the per-epoch history (epoch 0 = the base model's validation loss)."""
    from peft import LoraConfig, get_peft_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ocr = K.OCR(args.preset, gpu=0, **eval_overrides(args))
    model = ocr.model
    build, end = make_builder(ocr)

    # Sanity check: the supervised tokens must be exactly the target text (+ end-of-turn marker)
    sample = build(frame.img.iloc[0], frame.text.iloc[0])
    supervised = sample["input_ids"][0][sample["labels"][0] != -100]
    print(f"[{name}] end-of-turn {end!r} | seq len {sample['input_ids'].shape[1]} | inputs {list(sample.keys())}")
    print(f"[{name}] SUPERVISED: {ocr.tok.decode(supervised[:30])} ... {ocr.tok.decode(supervised[-6:])!r}")
    print(f"[{name}] TARGET    : {frame.text.iloc[0][:60]}")

    targets = [
        n for n, m in model.named_modules()
        if n.split(".")[-1] in LORA_TARGETS and "language_model" in n and hasattr(m, "weight")
    ]
    if not targets:
        raise RuntimeError("No LoRA target modules found; inspect [n for n, _ in model.named_modules()]")

    # prepare_model_for_kbit_training is deliberately not used: it casts every non-quantized weight
    # to fp32, which doubles Gemma's 5.6 GB per-layer embedding table and runs out of memory.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=targets, task_type="CAUSAL_LM"))

    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # fp32 master weights for GradScaler
    model.print_trainable_parameters()
    print(f"[{name}] {len(targets)} LoRA targets, GPU0 {torch.cuda.memory_allocated(0) / 2**30:.1f} GB")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = args.epochs * math.ceil(len(frame) / args.accum)
    warmup = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    scaler = torch.amp.GradScaler("cuda")

    def step_loss(image_path, text):
        # Gemma's vocabulary is 262k: logits for the full sequence (+ fp32 copy + grads) take several GB.
        # The supervised tokens are a suffix of the sequence, so only those positions get logits.
        enc = build(image_path, text)
        if enc["input_ids"].shape[1] > args.max_seq:
            return None
        labels = enc.pop("labels")
        n = int((labels[0] != -100).sum())  # supervised suffix length
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**{k: v.to("cuda:0") for k, v in enc.items()},
                           use_cache=False, logits_to_keep=n + 1).logits
        logits = logits[:, -(n + 1):-1].float()  # position t predicts token t+1
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels[:, -n:].reshape(-1).to(logits.device))

    @torch.no_grad()
    def val_loss():
        model.eval()
        losses = []
        for r in val_df.itertuples():
            loss = step_loss(r.img, r.text)
            if loss is not None and torch.isfinite(loss):
                losses.append(loss.item())
        return sum(losses) / max(1, len(losses))

    t0, history = time.time(), []

    # LoRA starts as a no-op, so this is the base model's validation loss
    vl = val_loss()
    history.append({"run": name, "epoch": 0, "train_loss": float("nan"), "val_loss": vl,
                    "skipped": 0, "minutes": 0.0})
    print(f"[{name}] epoch 0 (base)  val {vl:.4f}")

    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(frame)))
        random.shuffle(order)
        running, counted, skipped, pending = 0.0, 0, 0, 0

        for i, j in enumerate(order):
            r = frame.iloc[j]
            oom = False
            try:
                loss = step_loss(r.img, r.text)
                if loss is None or not torch.isfinite(loss):
                    skipped += 1
                    print("skip", j, "too long" if loss is None else "non-finite loss")
                else:
                    scaler.scale(loss / args.accum).backward()
                    running += loss.item()
                    counted += 1
                    pending += 1
            except torch.cuda.OutOfMemoryError:
                oom = True
            loss = None

            if oom:  # clean up outside `except`, where the tensors are released
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                gc.collect()
                torch.cuda.empty_cache()
                print(f"OOM on example {j} (seq {build(r.img, r.text)['input_ids'].shape[1]}) -> skipped")

            if epoch == 0 and i == 0:
                print(f"[{name}] peak GPU0 after first step: {torch.cuda.max_memory_allocated(0) / 2**30:.1f} GB")

            if pending and ((i + 1) % args.accum == 0 or i == len(order) - 1):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0

            if (i + 1) % 20 == 0:
                print(f"  [{name}] ep{epoch + 1} {i + 1}/{len(order)}  loss {running / max(1, counted):.4f}  "
                      f"{(time.time() - t0) / 60:.0f} min", flush=True)

        vl = val_loss()
        history.append({
            "run": name, "epoch": epoch + 1, "train_loss": running / max(1, counted), "val_loss": vl,
            "skipped": skipped, "minutes": round((time.time() - t0) / 60, 1),
        })
        print(f"[{name}] epoch {epoch + 1}/{args.epochs}  train {running / max(1, counted):.4f}  "
              f"val {vl:.4f}  skipped {skipped}  {(time.time() - t0) / 60:.0f} min")
        model.save_pretrained(f"lora_{name}_ep{epoch + 1}")

    del model, ocr, optimizer, params, scaler, scheduler, sample
    gc.collect()
    torch.cuda.empty_cache()  # inference.run() needs both GPUs free
    return history


def val_cer(tag):
    """Corpus CER, median page CER and number of pages with CER > 1 of results_<tag>.csv."""
    from jiwer import cer

    d = pd.read_csv(f"results_{tag}.csv").dropna(subset=["cer"])
    corpus = cer(list(d.ref_norm.astype(str)), list(d.pred_norm.fillna("").astype(str)))
    return corpus, d.cer.median(), int((d.cer > 1).sum())


def score_epochs(name, val_csv, root, args):
    """Score every epoch's adapter by CER on the validation pages; copy the best to lora_<name>_best."""
    scores = []
    for epoch in (args.eval_epochs or range(1, args.epochs + 1)):
        tag = f"{args.preset}-val-lora-{name}-ep{epoch}"
        K.run(args.preset, csv=str(val_csv), root=str(root), tag=tag,
              adapter=os.path.abspath(f"lora_{name}_ep{epoch}"), **eval_overrides(args))
        corpus, median, loops = val_cer(tag)
        scores.append({"run": name, "epoch": epoch, "corpus_cer": corpus,
                       "median_cer": median, "pages_cer>1": loops})
        print(f"[{name}] ep{epoch}: corpus CER {corpus:.4f}  median {median:.4f}  pages>1 {loops}")

    best = min(scores, key=lambda s: s["corpus_cer"])
    shutil.rmtree(f"lora_{name}_best", ignore_errors=True)
    shutil.copytree(f"lora_{name}_ep{best['epoch']}", f"lora_{name}_best")
    print(f"[{name}] best epoch {best['epoch']} -> lora_{name}_best")
    return scores


# ================= MAIN =================

def finetune(args):
    csv_path = Path(args.csv).resolve()
    root = Path(args.root).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)  # inference.run() / compare() read and write results_*.csv here

    random.seed(args.seed)
    train_df, val_df = load_namaa(csv_path, root, args.split, args.val_per_doc, args.val_frac, args.seed)

    # What inference.run() reads, plus a record of the split
    val_csv = output_dir / "val.csv"
    val_df[["path", "main_text"]].to_csv(val_csv, index=False, encoding="utf-8-sig")
    pd.concat([
        train_df[["img", "text", "doc"]].assign(split="train"),
        val_df[["img", "text", "doc"]].assign(split="val"),
    ]).to_csv("split.csv", index=False, encoding="utf-8-sig")

    if not args.no_baseline:
        K.run(args.preset, csv=str(val_csv), root=str(root), tag=f"{args.preset}-val-base", **eval_overrides(args))

    t0 = time.time()
    histories = train_run("namaa", train_df[["img", "text", "doc"]], val_df, args)
    pd.DataFrame(histories).to_csv("train_history.csv", index=False)
    scores = score_epochs("namaa", val_csv, root, args)
    pd.DataFrame(scores).to_csv("epoch_scores.csv", index=False)
    print(f"=== done, total {(time.time() - t0) / 60:.0f} min ===\n")

    merged = pd.DataFrame(histories).merge(pd.DataFrame(scores), on=["run", "epoch"], how="left")
    print(merged.to_string(index=False))
    if os.path.exists(f"results_{args.preset}-val-base.csv"):
        corpus, median, loops = val_cer(f"{args.preset}-val-base")
        print(f"baseline: corpus CER {corpus:.4f}  median {median:.4f}  pages>1 {loops}")
    print("\nOnly validation CER is meaningful; training pages were seen during training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of a vision-language OCR model on the NAMAA manuscript pages."
    )

    # Data
    parser.add_argument("--csv", default="dataset_relative.csv",
                        help="NAMAA CSV with `path` and `main_text` columns (default: dataset_relative.csv)")
    parser.add_argument("--root", default=".", help="Directory the image paths are relative to (default: .)")
    parser.add_argument("-o", "--output", default="lora_output", help="Output directory (default: lora_output)")
    parser.add_argument("--split", default="page", choices=["page", "doc"],
                        help="page: --val_per_doc pages from every group; doc: whole groups held out (default: page)")
    parser.add_argument("--val_per_doc", type=int, default=4,
                        help="Validation pages per group with --split page (default: 4)")
    parser.add_argument("--val_frac", type=float, default=0.25,
                        help="Share of NAMAA pages held out with --split doc (default: 0.25)")
    parser.add_argument("--max_seq", type=int, default=4096,
                        help="Skip examples longer than this many tokens (default: 4096)")

    # Model
    parser.add_argument("--preset", default="gemma4-e4b", choices=sorted(K.PRESETS),
                        help="Model preset from inference.py (default: gemma4-e4b)")
    parser.add_argument("--quant", default="4bit", choices=["none", "4bit", "8bit"], help="(default: 4bit)")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"],
                        help="Use fp32 if the loss keeps going non-finite (default: fp16)")
    parser.add_argument("--image_tokens", type=int, default=1120,
                        help="Gemma 4 visual token budget; 560 if training runs out of memory (default: 1120)")

    # Training (2e-4 / 3 epochs / accum 8 gave only ~12 updates on NAMAA and hurt CER)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--epochs", type=int, default=4, help="(default: 4)")
    parser.add_argument("--accum", type=int, default=2, help="Gradient accumulation steps, batch size 1 (default: 2)")
    parser.add_argument("--eval_epochs", type=int, nargs="+",
                        help="Epochs whose adapter is scored by CER (default: all)")
    parser.add_argument("--lora_r", type=int, default=16, help="(default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="(default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="(default: 0.05)")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip the baseline CER (only if results_<preset>-val-base.csv exists for this split)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    finetune(parser.parse_args())
