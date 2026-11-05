#!/usr/bin/env python3
"""
Quantize an Apertus checkpoint (e.g. the output of export_apertus15_text.py) to 4-bit weights (W4A16, group size
128) with AutoRound, for running the worker on GPUs that cannot hold the bf16 model (8B bf16 needs ~16 GB).

AutoRound tunes the rounding of each transformer block on calibration data. Unlike AWQ, it does not rescale
weights across the MLP activation, which is not possible with Apertus' xIELU activation.

Calibration samples should look like real worker inputs, so they are built with the worker's own prompt template
from config.yaml. The calibration file is JSONL with one translation example per line:

    {"src_lang": "Estonian", "tgt_lang": "English", "src_segm": "Tere!", "tgt_segm": "Hello!"}

Language names are used as-is in the prompt (the same names as in config.yaml `language_codes`). Use a sample of
the model's own training data, a few hundred examples covering all translation directions is enough.

Usage:
    python scripts/quantize_autoround.py MODEL_DIR OUT_DIR --calibration calib.jsonl [--config config/config.yaml]
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path

import yaml
from auto_round import AutoRound
from transformers import AutoTokenizer


def build_samples(examples, template, tokenizer, seqlen, nsamples, seed):
    """
    Pack prompt + translation examples into strings of at least `seqlen` tokens; AutoRound skips shorter samples.
    """
    random.Random(seed).shuffle(examples)
    texts = [template.format(src_lang=e["src_lang"], tgt_lang=e["tgt_lang"], text=e["src_segm"].strip())
             + e["tgt_segm"].strip() + tokenizer.eos_token for e in examples]

    samples, current, current_len = [], [], 0
    for text in texts:
        current.append(text)
        current_len += len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if current_len >= seqlen:
            samples.append("".join(current))
            current, current_len = [], 0
            if len(samples) == nsamples:
                break
    return samples


def restore_source_files(model: Path, out: Path):
    """
    AutoRound saves into a descriptive subdirectory (OUT_DIR/<model>-w4g128) and re-saves the configs and the
    tokenizer with the installed transformers version, which drops the transformers 4 RoPE keys, changes the
    tokenizer class and enables sampling in the generation config. Move the files to OUT_DIR and keep the source files,
    adding only the quantization config.
    """
    for config_file in out.glob("*/config.json"):
        for item in config_file.parent.iterdir():
            item.replace(out / item.name)
        config_file.parent.rmdir()

    quantization_config = json.loads((out / "config.json").read_text(encoding="utf-8"))["quantization_config"]
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    config["quantization_config"] = quantization_config
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for name in ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "chat_template.jinja", "LICENSE.txt"]:
        if (model / name).exists():
            shutil.copy2(model / name, out / name)

    # the weights are written as owner-only files, which e.g. the worker container user cannot read
    umask = os.umask(0)
    os.umask(umask)
    for item in out.iterdir():
        os.chmod(item, 0o666 & ~umask)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="bf16 model directory (ApertusForCausalLM)")
    parser.add_argument("out", type=Path, help="Output directory for the quantized model")
    parser.add_argument("--calibration", type=Path, required=True, help="JSONL file with translation examples")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"),
                        help="Worker config, the prompt template is taken from here")
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--iters", type=int, default=200, help="AutoRound tuning steps per block")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        template = yaml.safe_load(f)["model"]["prompt_template"]
    with open(args.calibration, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = build_samples(examples, template, tokenizer, args.seqlen, args.nsamples, args.seed)
    if len(samples) < args.nsamples:
        print(f"Warning: only {len(samples)} calibration samples of {args.seqlen} tokens "
              f"({len(examples)} examples), {args.nsamples} requested")

    autoround = AutoRound(
        str(args.model),
        tokenizer=tokenizer,
        scheme="W4A16",  # symmetric int4, group size 128; lm_head and embeddings stay in bf16
        dataset=samples,
        nsamples=len(samples),
        seqlen=args.seqlen,
        iters=args.iters,
        seed=args.seed,
    )
    autoround.quantize_and_save(str(args.out), format="auto_round")
    restore_source_files(args.model, args.out)

    metadata = {
        "source": str(args.model),
        "method": "AutoRound",
        "scheme": "W4A16",
        "calibration": str(args.calibration),
        "calibration_samples": len(samples),
        "seqlen": args.seqlen,
        "iters": args.iters,
    }
    (args.out / "quantization.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Quantized model saved to {args.out}")


if __name__ == "__main__":
    main()
