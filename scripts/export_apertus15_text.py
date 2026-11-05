#!/usr/bin/env python3
"""
Export the text-only part of an Apertus 1.5 checkpoint as a plain Apertus (v1) `ApertusForCausalLM` model.

Apertus 1.5 (`Apertus1p5ForConditionalGeneration`) is an Apertus 1 language backbone with discrete image and audio
tokenizers whose codes are appended to the vocabulary (ids >= 131072). For text-only input, the model is exactly:

    embed_tokens[:131072] -> ApertusModel -> lm_head (131072 rows)

so dropping the image/audio tokenizers, the multimodal embedding rows and the multimodal added tokens yields a
checkpoint that runs unchanged in stock transformers, vLLM and llama.cpp, which already support Apertus 1.

Usage:
    python scripts/export_apertus15_text.py SRC_DIR OUT_DIR

SRC_DIR is a local Apertus 1.5 checkpoint (e.g. `hf download swiss-ai/Apertus-v1.5-8B --local-dir SRC_DIR`),
including fine-tuned checkpoints saved in the same format.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TEXT_PREFIX = "model.language_model."
MAX_SHARD_BYTES = 4 * 1024 ** 3
UMASK = os.umask(0)
os.umask(UMASK)

TOKENIZER_CONFIG_KEYS = [
    "add_prefix_space", "bos_token", "eos_token", "pad_token", "unk_token", "clean_up_tokenization_spaces",
    "model_input_names", "padding_side", "tokenizer_class",
]


def text_vocab_size(src: Path, text_config: dict) -> int:
    """The text vocabulary is the prefix of the extended vocabulary that the pruned LM head covers."""
    vocab_size = text_config.get("output_vocab_size") or text_config["vocab_size"]
    tokenizer_config = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    base_vocab_size = tokenizer_config.get("base_vocab_size", vocab_size)
    if base_vocab_size != vocab_size:
        raise ValueError(f"LM head size ({vocab_size}) differs from the base tokenizer vocabulary ({base_vocab_size})")
    return vocab_size


def export_weights(src: Path, out: Path, vocab_size: int):
    weight_map = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    names = sorted(name for name in weight_map if name.startswith(TEXT_PREFIX) or name == "lm_head.weight")

    shards, current, current_bytes = [], {}, 0
    for shard_file in sorted({weight_map[name] for name in names}):
        with safe_open(src / shard_file, framework="pt") as f:
            for name in (n for n in names if weight_map[n] == shard_file):
                tensor = f.get_tensor(name)
                new_name = "model." + name[len(TEXT_PREFIX):] if name.startswith(TEXT_PREFIX) else name
                if new_name in ("model.embed_tokens.weight", "lm_head.weight"):
                    if tensor.shape[0] < vocab_size:
                        raise ValueError(f"{name} has {tensor.shape[0]} rows, expected at least {vocab_size}")
                    tensor = tensor[:vocab_size].clone()
                tensor_bytes = tensor.numel() * tensor.element_size()
                if current and current_bytes + tensor_bytes > MAX_SHARD_BYTES:
                    shards.append(current)
                    current, current_bytes = {}, 0
                current[new_name] = tensor.contiguous()
                current_bytes += tensor_bytes
    shards.append(current)

    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    for i, shard in enumerate(shards, start=1):
        shard_file = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, out / shard_file, metadata={"format": "pt"})
        # safetensors writes owner-only files, which e.g. the worker container user cannot read
        os.chmod(out / shard_file, 0o666 & ~UMASK)
        for name, tensor in shard.items():
            index["weight_map"][name] = shard_file
            index["metadata"]["total_size"] += tensor.numel() * tensor.element_size()
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return len(index["weight_map"])


def export_config(src: Path, out: Path, src_config: dict, vocab_size: int):
    text_config = src_config["text_config"]
    rope = dict(text_config["rope_parameters"])
    rope_theta = rope.pop("rope_theta")
    config = {
        "architectures": ["ApertusForCausalLM"],
        "model_type": "apertus",
        "vocab_size": vocab_size,
        # transformers < 5 reads `rope_theta` + `rope_scaling`, transformers >= 5 reads `rope_parameters`
        "rope_theta": rope_theta,
        "rope_scaling": {**rope, "type": rope["rope_type"]},
        "rope_parameters": {**rope, "rope_theta": rope_theta},
    }
    for key in ["attention_bias", "attention_dropout", "bos_token_id", "eos_token_id", "pad_token_id", "hidden_act",
                "hidden_dropout", "hidden_size", "initializer_range", "intermediate_size", "max_position_embeddings",
                "mlp_bias", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "post_norm", "qk_norm",
                "rms_norm_eps", "tie_word_embeddings"]:
        if key in text_config:
            config[key] = text_config[key]
    config["dtype"] = text_config.get("dtype", "bfloat16")
    config["use_cache"] = True
    # without a version, transformers assumes an old Mistral-converted tokenizer and warns about its regex
    config["transformers_version"] = src_config["transformers_version"]
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    generation_config = json.loads((src / "generation_config.json").read_text(encoding="utf-8"))
    generation_config.pop("transformers_version", None)
    (out / "generation_config.json").write_text(json.dumps(generation_config, indent=2) + "\n", encoding="utf-8")


def export_tokenizer(src: Path, out: Path, vocab_size: int, max_length: int):
    tokenizer = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    if max(tokenizer["model"]["vocab"].values()) >= vocab_size:
        raise ValueError("The base BPE vocabulary uses ids outside of the text vocabulary")
    # drop image/audio code tokens so that they can never be produced from user text
    tokenizer["added_tokens"] = [t for t in tokenizer["added_tokens"] if t["id"] < vocab_size]
    (out / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False) + "\n", encoding="utf-8")

    src_config = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    config = {key: src_config[key] for key in TOKENIZER_CONFIG_KEYS if key in src_config}
    config["model_max_length"] = max_length
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for name in ["special_tokens_map.json", "chat_template.jinja", "LICENSE.txt"]:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="Apertus 1.5 checkpoint directory")
    parser.add_argument("out", type=Path, help="Output directory for the text-only ApertusForCausalLM checkpoint")
    args = parser.parse_args()

    src_config = json.loads((args.src / "config.json").read_text(encoding="utf-8"))
    if src_config.get("model_type") != "apertus1p5":
        raise ValueError(f"Expected an Apertus 1.5 checkpoint, got model_type={src_config.get('model_type')}")
    text_config = src_config["text_config"]
    vocab_size = text_vocab_size(args.src, text_config)

    args.out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        tensor_count = export_weights(args.src, args.out, vocab_size)
    export_config(args.src, args.out, src_config, vocab_size)
    export_tokenizer(args.src, args.out, vocab_size, text_config["max_position_embeddings"])
    print(f"Exported {tensor_count} tensors with a {vocab_size}-token vocabulary to {args.out}")


if __name__ == "__main__":
    main()
