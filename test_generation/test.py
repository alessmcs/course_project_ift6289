import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)


SYSTEM_PROMPT = """You are a software engineer.
Answer the provided question using only the provided code and documentation.
Do not invent things that are not present.
Be technical, precise, and concise.
Do not restate the question."""


def build_user_prompt(code: str, documentation: str, question: str) -> str:
    return f"""Code:
{code}

Documentation:
{documentation}

Question:
{question}

Answer directly and technically."""


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_model_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def load_model(base_model_name: str, adapter_path: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )

    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=False,
    )
    model.eval()

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None:
            model.generation_config.bos_token_id = tokenizer.bos_token_id

    return model, tokenizer


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    code: str,
    documentation: str,
    question: str,
    max_new_tokens: int,
):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(code, documentation, question),
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    enc = tokenizer(prompt_text, return_tensors="pt")
    device = get_model_input_device(model)
    enc = {k: v.to(device) for k, v in enc.items()}

    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"])

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    outputs = model.generate(
        **enc,
        generation_config=gen_config,
    )

    new_tokens = outputs[:, enc["input_ids"].shape[1]:]
    text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="/data/rech/jaouaime/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="outputs/llama31_8b_dpo/checkpoint-800",
    )
    parser.add_argument("--doc_key", type=str, default="text")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.base_model_name, args.adapter_path)

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(read_jsonl(args.test_file), start=1):
            code = str(ex.get("code") or "").strip()
            documentation = str(ex.get(args.doc_key) or "").strip()
            question = str(ex.get("question") or "").strip()

            pred = generate_one(
                model=model,
                tokenizer=tokenizer,
                code=code,
                documentation=documentation,
                question=question,
                max_new_tokens=args.max_new_tokens,
            )

            row = dict(ex)
            row["prediction"] = pred
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            if i % 50 == 0:
                print(f"Done {i} examples")

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()