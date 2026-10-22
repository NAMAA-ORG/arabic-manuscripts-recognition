# pip install -U torch transformers accelerate peft bitsandbytes pillow pandas jiwer

"""Transcribe manuscript pages with vision-language models and score them by CER.

Built for Kaggle (2x T4) but works on any CUDA machine. A benchmark run puts one
model copy on each GPU and splits the pages between them (one worker process
per GPU), which roughly doubles throughput.

Shell:
    python inference_multiple_vlms.py --list
    python inference_multiple_vlms.py --preset qari --csv dataset_relative.csv --root .
    python inference_multiple_vlms.py --preset gemma4-e4b --quant 4bit --device_map per-gpu --tag gemma4-e4b-4bit
    python inference_multiple_vlms.py --compare

Python:
    import inference_multiple_vlms as K
    ocr = K.OCR("qari", gpu=0); print(ocr.transcribe("page.png")); ocr.free()
    K.run("qari", csv="dataset_relative.csv", root=".")
    K.compare()

T4 has no bf16 or FlashAttention-2, so models run in fp16 with sdpa attention.
Models too big for one T4: --quant 4bit (still one copy per GPU) or
--device_map auto (one copy split over all GPUs).
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import unicodedata
import warnings
from types import SimpleNamespace

import pandas as pd


DEFAULT_PROMPT = "Transcribe the Arabic manuscript exactly. Output only the main text, no explanation."

# Prompts follow each model card. Any field can be overridden per call or on the command line.
PRESETS = {
    "qwen25-3b":     dict(model="Qwen/Qwen2.5-VL-3B-Instruct"),
    "qwen25-7b":     dict(model="Qwen/Qwen2.5-VL-7B-Instruct", quant="4bit"),
    "qwen3-4b":      dict(model="Qwen/Qwen3-VL-4B-Instruct"),
    "qwen3-8b":      dict(model="Qwen/Qwen3-VL-8B-Instruct", quant="4bit"),
    "qari":          dict(model="NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
                          prompt="Below is the image of one page of a document, as well as some raw textual "
                                 "content that was previously extracted for it. Just return the plain text "
                                 "representation of this document as if you were reading it naturally. "
                                 "Do not hallucinate."),
    "baseer-nakba":  dict(model="Misraj/Baseer__Nakba",  # historical Arabic HTR (NAKBA 2026 winner)
                          prompt="Extract the text from the above document.",
                          max_pixels=1280 * 28 * 28),
    "internvl35-4b": dict(model="OpenGVLab/InternVL3_5-4B-HF"),
    "nanonets":      dict(model="nanonets/Nanonets-OCR2-3B",
                          prompt="Extract the text from the above document as if you were reading it naturally."),
    "dots":          dict(model="dots-studio/dots.ocr", loader="causal", trust_remote_code=True,  # experimental on T4
                          prompt="Extract the text content from this image."),
    # Gemma 4: image_tokens = visual token budget (70/140/280/560/1120; the card recommends high for OCR).
    # 1120 tokens is about 2.6 MP, so max_pixels is raised to match. If fp16 overflows on T4, use --dtype fp32.
    # attn="auto": the audio tower has no sdpa, so transformers picks per submodule.
    "gemma4-e2b":    dict(model="google/gemma-4-E2B-it", loader="multimodal", image_tokens=1120, attn="auto",
                          max_pixels=1120 * 9 * 16 * 16),
    "gemma4-e4b":    dict(model="google/gemma-4-E4B-it", loader="multimodal", image_tokens=1120, attn="auto",
                          max_pixels=1120 * 9 * 16 * 16, device_map="auto"),  # 16 GB in fp16: split over both T4s
    "gemma4-12b":    dict(model="google/gemma-4-12B-it", loader="multimodal", image_tokens=1120, attn="auto",
                          max_pixels=1120 * 9 * 16 * 16, quant="4bit"),
    "gemma4-26b":    dict(model="google/gemma-4-26B-A4B-it", loader="multimodal", image_tokens=1120, attn="auto",
                          max_pixels=1120 * 9 * 16 * 16, quant="4bit", device_map="auto"),  # MoE, ~52 GB download
}

DEFAULTS = dict(
    prompt=DEFAULT_PROMPT,
    loader="vlm",              # vlm | multimodal | causal
    quant="none",              # none | 4bit | 8bit
    dtype="fp16",              # fp16 | fp32
    device_map="per-gpu",      # per-gpu: one copy per GPU | auto: one copy split over all GPUs
    max_pixels=1024 * 28 * 28,  # ~0.8 MP; lower if out of memory
    max_new_tokens=1536,
    trust_remote_code=False,
    image_tokens=None,         # Gemma 4 visual token budget; None = model default
    attn="sdpa",               # auto = let transformers choose (sdpa where supported, else eager)
    adapter=None,              # LoRA adapter directory (peft) loaded on top of the model
)


def config(preset="qwen25-3b", **overrides):
    """Preset + defaults + overrides (None values are ignored). `model` may be any Hugging Face id."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")

    cfg = {**DEFAULTS, **PRESETS[preset], "tag": preset}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return SimpleNamespace(**cfg)


def list_presets():
    return pd.DataFrame([
        {"preset": name, **{f: getattr(config(name), f) for f in ("model", "quant", "device_map")}}
        for name in PRESETS
    ]).set_index("preset")


# ================= TEXT UTILITIES =================

def normalize_arabic(text: str) -> str:
    """Normalization applied to both reference and prediction before computing CER."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]", "", text)  # diacritics
    text = text.replace("\u0640", "")                                          # tatweel
    text = re.sub("[\u0625\u0623\u0622\u0671]", "\u0627", text)                # alef variants -> alef
    text = text.replace("\u0649", "\u064a")                                    # alef maqsura -> ya
    text = text.replace("\u0629", "\u0647")                                    # ta marbuta -> ha
    return re.sub(r"\s+", " ", text).strip()


def collapse_repeats(text: str, max_rep: int = 2) -> str:
    """Cut runaway loops: a 1-8 word phrase repeated more than max_rep times in a row is kept max_rep times."""
    pattern = r"(\b\S+(?:\s+\S+){0,7}?)(?:\s+\1\b){%d,}" % max_rep
    return re.sub(pattern, lambda m: " ".join([m.group(1)] * max_rep), text)


def cap_pixels(image, max_pixels: int):
    """Downscale so width * height <= max_pixels (processors differ in their own pixel options)."""
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    return image.resize((max(28, int(w * scale)), max(28, int(h * scale))))


def page_path(root, rel):
    """dataset_relative.csv stores Windows-style paths; make them work on Linux too."""
    return os.path.join(root, str(rel).replace("\\", "/"))


# ================= MODEL =================

class OCR:
    """One loaded model. gpu=0/1 pins it to one GPU; device_map="auto" spreads it over all visible GPUs."""

    def __init__(self, preset="qwen25-3b", gpu=0, **overrides):
        import torch
        import transformers
        from packaging.version import Version
        from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                                  AutoProcessor, BitsAndBytesConfig)

        self.cfg = c = config(preset, **overrides)
        dtype = torch.float16 if c.dtype == "fp16" else torch.float32

        # bitsandbytes 8-bit warns on every fp32 -> fp16 cast (harmless)
        warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")

        dtype_kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
        kw = dict(trust_remote_code=c.trust_remote_code, **{dtype_kw: dtype},
                  device_map="auto" if c.device_map == "auto" else {"": gpu})
        if c.attn != "auto":
            kw["attn_implementation"] = c.attn
        if c.quant == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        elif c.quant == "8bit":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        model_cls = {
            "causal": AutoModelForCausalLM,
            "multimodal": getattr(transformers, "AutoModelForMultimodalLM", AutoModelForImageTextToText),
        }.get(c.loader, AutoModelForImageTextToText)

        self.model = model_cls.from_pretrained(c.model, **kw).eval()
        if c.adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, c.adapter).eval()

        self.processor = AutoProcessor.from_pretrained(c.model, trust_remote_code=c.trust_remote_code)
        if c.image_tokens:
            # Gemma 4 keeps the budget on both the processor and its image processor
            for obj in (self.processor, getattr(self.processor, "image_processor", None)):
                for attr in ("max_soft_tokens", "image_seq_length"):
                    if obj is not None and hasattr(obj, attr):
                        setattr(obj, attr, int(c.image_tokens))
        self.tok = getattr(self.processor, "tokenizer", self.processor)

    def transcribe(self, image, prompt=None, max_new_tokens=None, max_pixels=None, **gen_kwargs):
        """image: PIL image or file path. Extra kwargs go to model.generate (e.g. repetition_penalty=1.1)."""
        import torch
        from PIL import Image

        c = self.cfg
        if not isinstance(image, Image.Image):
            image = Image.open(str(image).replace("\\", "/"))
        image = cap_pixels(image.convert("RGB"), max_pixels or c.max_pixels)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt or c.prompt},
        ]}]
        # enable_thinking is ignored by chat templates without a thinking mode
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)

        gen = dict(
            max_new_tokens=max_new_tokens or c.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,   # mild; higher hurts Arabic
            no_repeat_ngram_size=12,   # blocks long verbatim loops only
        )
        gen.update(gen_kwargs)

        try:
            with torch.inference_mode():
                out = self.model.generate(**inputs, **gen)
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            return self.tok.decode(new_tokens, skip_special_tokens=True).strip()
        finally:
            del inputs
            torch.cuda.empty_cache()

    def free(self):
        """Release GPU memory (do this before run(), which needs all GPUs)."""
        import gc
        import torch

        del self.model, self.processor, self.tok
        gc.collect()
        torch.cuda.empty_cache()


# ================= BENCHMARK =================

def _worker(args):
    """One GPU's share of the pages (started by run() as a subprocess)."""
    df = pd.read_csv(args.csv).iloc[args.shard::args.nshards]
    overrides = {k: getattr(args, k) for k in DEFAULTS if k != "trust_remote_code"}
    ocr = OCR(args.preset, gpu=0, model=args.model,
              trust_remote_code=args.trust_remote_code or None, **overrides)

    rows = []
    for i, row in df.iterrows():
        pred, err = "", ""
        try:
            pred = ocr.transcribe(page_path(args.root, row["path"]))
        except Exception as exc:  # keep going on out-of-memory / missing file
            err = repr(exc)
            print(f"[gpu{args.shard}] row {i} failed: {err}", flush=True)
        rows.append({"idx": i, "raw_prediction": pred, "error": err})
        print(f"[gpu{args.shard}] {len(rows)}/{len(df)} done", flush=True)

    pd.DataFrame(rows).to_csv(f"shard_{args.tag}_{args.shard}.csv", index=False, encoding="utf-8-sig")


def score(df):
    """Adds prediction / ref_norm / pred_norm / cer columns and prints corpus and per-page CER."""
    from jiwer import cer

    df["prediction"] = df["raw_prediction"].fillna("").astype(str).map(collapse_repeats)
    df["ref_norm"] = df["main_text"].astype(str).map(normalize_arabic)
    df["pred_norm"] = df["prediction"].map(normalize_arabic)

    ok = df[(df["error"].fillna("") == "") & (df["ref_norm"].str.len() > 0)].copy()
    ok["cer"] = [cer(r, p) if p else 1.0 for r, p in zip(ok["ref_norm"], ok["pred_norm"])]
    df["cer"] = ok["cer"]

    print(f"Pages scored: {len(ok)}/{len(df)}")
    print(f"Corpus CER (normalized):   {cer(list(ok.ref_norm), list(ok.pred_norm)):.4f}")
    print(f"Mean per-page CER:         {ok.cer.mean():.4f}   median: {ok.cer.median():.4f}")
    print(f"Pages with CER > 1 (loops): {(ok.cer > 1).sum()}")
    return df


def run(preset="qwen25-3b", csv="dataset_relative.csv", root=".", nshards=2, limit=None, **overrides):
    """Benchmark a preset on the CSV (or its first `limit` rows), one worker process per GPU.

    Writes results_<tag>.csv to the working directory and returns it as a DataFrame.
    Free any OCR() loaded in the same process first.
    """
    c = config(preset, **overrides)
    if c.device_map == "auto":
        nshards = 1

    if limit:
        subset = f"_subset_{c.tag}.csv"
        pd.read_csv(csv).head(limit).to_csv(subset, index=False, encoding="utf-8-sig")
        csv = subset

    procs = []
    for shard in range(nshards):
        env = dict(os.environ)
        if c.device_map != "auto":
            env["CUDA_VISIBLE_DEVICES"] = str(shard)

        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--preset", preset,
               "--shard", str(shard), "--nshards", str(nshards), "--csv", csv, "--root", root]
        for k in ("model", "tag", *DEFAULTS):
            v = getattr(c, k)
            if k == "trust_remote_code":
                cmd += ["--trust_remote_code"] if v else []
            elif v is not None:
                cmd += [f"--{k}", str(v)]
        procs.append(subprocess.Popen(cmd, env=env))

    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"Worker exit codes {codes}; see the log above")

    df = pd.read_csv(csv)
    res = pd.concat([
        pd.read_csv(f"shard_{c.tag}_{s}.csv", keep_default_na=False) for s in range(nshards)
    ]).set_index("idx")

    print(f"\n[{c.tag}] {c.model}  quant={c.quant}  device_map={c.device_map}  -> results_{c.tag}.csv")
    df = score(df.join(res))
    df.to_csv(f"results_{c.tag}.csv", index=False, encoding="utf-8-sig")
    return df


def compare(pattern="results_*.csv"):
    """CER summary of every results_<tag>.csv in the working directory, best first."""
    from jiwer import cer

    rows = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f).dropna(subset=["cer"])
        rows.append({
            "run": os.path.basename(f)[len("results_"):-len(".csv")],
            "pages": len(d),
            "corpus_cer": cer(list(d.ref_norm.astype(str)), list(d.pred_norm.fillna("").astype(str))),
            "median_cer": d.cer.median(),
            "loops": int((d.cer > 1).sum()),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("corpus_cer").reset_index(drop=True)


# ================= COMMAND LINE =================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Transcribe manuscript pages with a vision-language model and score them by CER."
    )

    parser.add_argument("--list", action="store_true", help="Print the presets and exit")
    parser.add_argument("--compare", action="store_true",
                        help="Print the CER table of all results_*.csv in the working directory and exit")
    parser.add_argument("--preset", default="qwen25-3b", choices=sorted(PRESETS),
                        help="Model preset (default: qwen25-3b)")
    parser.add_argument("--csv", default="dataset_relative.csv",
                        help="CSV with `path` and `main_text` columns (default: dataset_relative.csv)")
    parser.add_argument("--root", default=".", help="Directory the image paths are relative to (default: .)")
    parser.add_argument("--nshards", type=int, default=2, help="Number of GPUs to split the pages over (default: 2)")
    parser.add_argument("--limit", type=int, help="Only the first N rows of the CSV")

    # Preset overrides; None means "take it from the preset"
    parser.add_argument("--model", help="Any Hugging Face model id")
    parser.add_argument("--tag", help="Name for the output files (default: the preset name)")
    parser.add_argument("--prompt")
    parser.add_argument("--loader", choices=["vlm", "causal", "multimodal"])
    parser.add_argument("--quant", choices=["none", "4bit", "8bit"])
    parser.add_argument("--dtype", choices=["fp16", "fp32"])
    parser.add_argument("--device_map", choices=["per-gpu", "auto"],
                        help="per-gpu: one copy per GPU; auto: one copy split across all GPUs")
    parser.add_argument("--max_pixels", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--image_tokens", type=int, help="Gemma 4 visual token budget (70-1120)")
    parser.add_argument("--attn", choices=["sdpa", "eager", "auto"])
    parser.add_argument("--adapter", help="LoRA adapter directory (peft)")
    parser.add_argument("--trust_remote_code", action="store_true", default=None)

    # Internal: used by run() to start one worker per GPU
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        print(list_presets().to_string())
    elif args.compare:
        print(compare().to_string())
    elif args.worker:
        _worker(args)
    else:
        run(
            args.preset,
            csv=args.csv,
            root=args.root,
            nshards=args.nshards,
            limit=args.limit,
            **{k: getattr(args, k) for k in ("model", "tag", *DEFAULTS)}
        )
