import gc
import os
import json
import re
import ast
import math
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from trl import (
    SFTTrainer,
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
)
from peft import LoraConfig, TaskType
from auto_gptq import AutoGPTQForCausalLM


# ============================================================
# Configuration
# ============================================================

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "/Tmp/raoufiho/NLP_Project/"
SFT_DIR    = os.path.join(OUTPUT_DIR, "sft_student")
PPO_DIR    = os.path.join(OUTPUT_DIR, "ppo_student")

TRAIN_JSON_PATH = "/Tmp/raoufiho/NLP_Project/datasets/train.jsonl"
VAL_JSON_PATH   = "/Tmp/raoufiho/NLP_Project/datasets/val.jsonl"
TEST_JSON_PATH  = "/Tmp/raoufiho/NLP_Project/datasets/test.jsonl"

LOCAL_TEACHER_MODEL_PATH = "/data/rech/raoufiho/Llama-3.3-70B-Instruct-GPTQ-Int4"

USE_4BIT = True
SEED     = 42

MAX_PROMPT_LEN  = 512
MAX_NEW_TOKENS  = 32

# PPO config
PPO_BATCH_SIZE      = 8    # reduced to fit GPU memory during step
PPO_MINI_BATCH_SIZE = 2
PPO_LR              = 5e-6
PPO_INIT_KL         = 0.05
PPO_EPOCHS          = 1

# Hybrid reward weights
CORRECTNESS_REWARD_WEIGHT = 0.6
TEACHER_REWARD_WEIGHT     = 0.4

# Teacher reward cache path
TEACHER_CACHE_PATH = os.path.join(OUTPUT_DIR, "teacher_rewards.json")

# ── GPU memory budgets ───────────────────────────────────────
#
# Phase SFT    (CUDA_VISIBLE_DEVICES=0):
#   GPU 0 only — student 4-bit training
#
# Phase CACHE  (CUDA_VISIBLE_DEVICES=0,1,2,3):
#   Teacher on GPUs 2 & 3
#
# Phase PPO    (CUDA_VISIBLE_DEVICES=0,1,2,3):
#   Teacher is NOT loaded — freed after caching
#   Student  on GPU 0  (~5 GiB weights + activations/gradients)
#   Ref      on GPU 1  (~5 GiB, frozen)
#   GPUs 2 & 3 free for overflow activations

TEACHER_MEM = {0: 0, 1: 0, 2: "23GiB", 3: "23GiB"}


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT + PPO training with Llama-3-8B student and Llama-3.3-70B teacher"
    )
    parser.add_argument(
        "--phase",
        choices=["sft", "cache", "ppo", "cache_and_ppo"],
        default="sft",
        help=(
            "Which phase to run:\n"
            "  sft           — supervised fine-tuning only\n"
            "                  Launch: CUDA_VISIBLE_DEVICES=0\n"
            "  cache         — pre-compute teacher rewards and save to disk\n"
            "                  Launch: CUDA_VISIBLE_DEVICES=0,1,2,3\n"
            "  ppo           — PPO refinement using cached teacher rewards\n"
            "                  Launch: CUDA_VISIBLE_DEVICES=0,1,2,3\n"
            "  cache_and_ppo — run cache then ppo in one process\n"
            "                  Launch: CUDA_VISIBLE_DEVICES=0,1,2,3\n"
        ),
    )
    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# GPU diagnostics
# ============================================================

def print_gpu_memory(label: str = ""):
    n = torch.cuda.device_count()
    print(f"\n── GPU memory {label} (visible devices: {n}) ──")
    for i in range(n):
        free, total = torch.cuda.mem_get_info(i)
        used = total - free
        print(
            f"  GPU {i}: {used/1024**3:.1f} GiB used / "
            f"{total/1024**3:.1f} GiB total  "
            f"({free/1024**3:.1f} GiB free)"
        )


def free_gpu_memory(*objects):
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ============================================================
# Dataset loading
# ============================================================

def clean_reasoning(text: str) -> str:
    text = str(text).strip()
    return re.sub(r"\s+", " ", text).strip()


def load_json_dataset(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"Dataset file is empty: {path}")

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError(f"Expected JSON array or object in {path}")
    except json.JSONDecodeError:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_num} in {path}: {e}"
                    ) from e
                data.append(item)

    cleaned_data = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} in {path} is not a JSON object")
        required_keys = {"code", "question", "answer"}
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Item {i} missing keys: {sorted(missing)}")
        documentation = item.get("documentation", item.get("reasoning", ""))
        cleaned_data.append({
            "code":          str(item["code"]),
            "documentation": clean_reasoning(documentation),
            "question":      str(item["question"]),
            "answer":        str(item["answer"]),
        })

    return cleaned_data


# ============================================================
# Prompt formatting
# ============================================================

def build_prompt(example: Dict) -> str:
    return (
        "You must answer using only the code and documentation.\n"
        "Return only the final answer. No explanation.\n\n"
        f"Code:\n{example['code']}\n\n"
        f"Documentation:\n{example['documentation']}\n\n"
        f"Question:\n{example['question']}\n\n"
        "Answer:\n"
    )


def build_sft_text(example: Dict) -> str:
    return build_prompt(example) + str(example["answer"]).strip()


# ============================================================
# Normalization / answer matching
# ============================================================

def strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def normalize_text(s: str) -> str:
    s = s.strip()
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    return re.sub(r"\s+", " ", s).strip().lower()


def try_literal_eval(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def canonicalize_answer(s: str):
    s = s.strip()
    for marker in ("Question:", "Code:", "Documentation:", "Answer:"):
        s = s.split(marker)[0]
    s = s.strip()

    lit = try_literal_eval(s)
    if lit is not None:
        return lit

    lower = normalize_text(s)
    if lower == "true":  return True
    if lower == "false": return False
    if lower == "none":  return None

    try:
        return float(s) if "." in s else int(s)
    except Exception:
        pass

    return normalize_text(strip_quotes(s))


def answers_match(pred: str, gold: str) -> bool:
    pred_c = canonicalize_answer(pred)
    gold_c = canonicalize_answer(gold)
    if pred_c == gold_c:
        return True
    if isinstance(pred_c, (int, float)) and isinstance(gold_c, (int, float)):
        return math.isclose(float(pred_c), float(gold_c), rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(pred_c, list) and isinstance(gold_c, list):
        try:
            return sorted(pred_c) == sorted(gold_c)
        except Exception:
            return pred_c == gold_c
    pred_n = normalize_text(str(pred_c))
    gold_n = normalize_text(str(gold_c))
    return pred_n == gold_n or gold_n in pred_n


def leakage_penalty(text: str) -> float:
    bad_markers = [
        "### system", "### code", "### documentation",
        "### question", "### answer",
        "system:", "question:", "documentation:", "code:",
        "http://", "https://", "source:",
    ]
    lower = text.lower()
    return 0.25 * sum(m in lower for m in bad_markers)


def length_penalty(text: str, max_reasonable_chars: int = 80) -> float:
    text = text.strip()
    if len(text) <= max_reasonable_chars:
        return 0.0
    return min(0.3, (len(text) - max_reasonable_chars) / 300.0)


def compute_correctness_reward(pred: str, gold: str) -> float:
    base   = 1.0 if answers_match(pred, gold) else -1.0
    reward = base - leakage_penalty(pred) - length_penalty(pred)
    return max(-1.0, min(1.0, reward))


# ============================================================
# Teacher reward (used only during cache phase)
# ============================================================

@torch.no_grad()
def compute_teacher_reward(
    prompt: str,
    response: str,
    tokenizer,
    model,
) -> float:
    full_text = prompt + response
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LEN + MAX_NEW_TOKENS,
        return_attention_mask=True,
    )
    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LEN,
        return_attention_mask=True,
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]
    device     = next(model.parameters()).device
    inputs     = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits  = outputs.logits.float()

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    labels = shift_labels.clone()
    labels[..., :prompt_len - 1] = -100

    loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels.view(-1),
    )
    return -loss.item()


def normalize_teacher_reward(r: float) -> float:
    return max(-5.0, min(0.0, r)) / 5.0


# ============================================================
# Teacher cache  —  compute and save to disk
# ============================================================

def precompute_teacher_rewards(
    train_data: List[Dict],
    teacher_tokenizer,
    teacher,
) -> Dict[int, float]:
    """
    Score every training example's gold answer under the teacher.
    Results are normalized and saved to TEACHER_CACHE_PATH.
    If the cache already exists, it is loaded instead.
    """
    if os.path.exists(TEACHER_CACHE_PATH):
        print(f"\nLoading cached teacher rewards from {TEACHER_CACHE_PATH}")
        with open(TEACHER_CACHE_PATH) as f:
            raw = json.load(f)
        return {int(k): float(v) for k, v in raw.items()}

    print(f"\nPre-computing teacher rewards for {len(train_data)} examples …")
    rewards: Dict[int, float] = {}

    for i, ex in enumerate(train_data):
        prompt = build_prompt(ex)
        r_raw  = compute_teacher_reward(
            prompt, str(ex["answer"]), teacher_tokenizer, teacher
        )
        rewards[i] = normalize_teacher_reward(r_raw)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(train_data)} done  "
                  f"(last normalized reward: {rewards[i]:.4f})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TEACHER_CACHE_PATH, "w") as f:
        json.dump({str(k): v for k, v in rewards.items()}, f, indent=2)
    print(f"Teacher rewards saved to {TEACHER_CACHE_PATH}")
    return rewards


def load_teacher_cache() -> Dict[int, float]:
    if not os.path.exists(TEACHER_CACHE_PATH):
        raise FileNotFoundError(
            f"Teacher reward cache not found at {TEACHER_CACHE_PATH}.\n"
            "Run --phase cache first."
        )
    with open(TEACHER_CACHE_PATH) as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


# ============================================================
# Teacher model loading (cache phase only)
# ============================================================

def load_teacher():
    print("\nLoading teacher tokenizer …")
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_TEACHER_MODEL_PATH,
        local_files_only=True,
    )
    print_gpu_memory("before teacher load")
    print("Loading teacher model onto GPUs 2 & 3 …")
    teacher = AutoGPTQForCausalLM.from_quantized(
        LOCAL_TEACHER_MODEL_PATH,
        local_files_only=True,
        device_map="auto",
        max_memory=TEACHER_MEM,
        use_safetensors=True,
        inject_fused_attention=False,
        low_cpu_mem_usage=True,
    )
    teacher.eval()
    print_gpu_memory("after teacher load")
    return teacher_tokenizer, teacher


def diagnose_teacher(tokenizer, model):
    first_device = next(model.parameters()).device
    inputs = tokenizer("Hello, how are you?", return_tensors="pt",
                       return_attention_mask=True)
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    print("Teacher first device:", first_device)
    print("Teacher dtype:",        next(model.parameters()).dtype)
    with torch.no_grad():
        logits = model(**inputs).logits.float()
    print("Logits shape:", logits.shape)
    print("Has nan:", torch.isnan(logits).any().item())
    print("Has inf:", torch.isinf(logits).any().item())


def sanity_check_teacher_reward(teacher_tokenizer, teacher, example: Dict):
    prompt = build_prompt(example)
    r1 = compute_teacher_reward(prompt, str(example["answer"]),
                                teacher_tokenizer, teacher)
    r2 = compute_teacher_reward(prompt, "I don't know.",
                                teacher_tokenizer, teacher)
    print(f"Teacher correct-answer reward: {r1:.4f}")
    print(f"Teacher wrong-answer reward:   {r2:.4f}")
    print("Sanity check PASSED" if r1 > r2 else "WARNING: sanity check failed")


# ============================================================
# Tokenizer / quantization / LoRA helpers
# ============================================================

def make_bnb_config():
    if not USE_4BIT:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def make_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


# ============================================================
# Evaluation helpers
# ============================================================

@torch.no_grad()
def generate_answer(model, tokenizer, prompt, device, max_new_tokens=MAX_NEW_TOKENS):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LEN,
        padding=False,
        return_attention_mask=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def evaluate_model(model, tokenizer, data, device, name):
    total  = min(50, len(data))
    exact  = 0
    leaked = 0
    print(f"\n===== Evaluation: {name} =====")
    for i, ex in enumerate(data[:total]):
        prompt = build_prompt(ex)
        pred   = generate_answer(model, tokenizer, prompt, device)
        if answers_match(pred, str(ex["answer"])):
            exact += 1
        if leakage_penalty(pred) > 0:
            leaked += 1
        if i < 5:
            print(f"\nExample {i+1}")
            print("Q:",    ex["question"])
            print("Gold:", ex["answer"])
            print("Pred:", pred)
    print(f"\n{name} exact-ish accuracy: {exact / total:.3f}")
    print(f"{name} leakage rate:       {leaked / total:.3f}")


# ============================================================
# SFT phase
# ============================================================
# Launch: CUDA_VISIBLE_DEVICES=0 python RL_traning.py --phase sft
# 4-bit training requires a single visible GPU due to Accelerate
# ============================================================

def run_sft(train_data: List[Dict], val_data: List[Dict]):
    n_visible = torch.cuda.device_count()
    if n_visible > 1:
        raise RuntimeError(
            f"SFT detected {n_visible} visible GPUs. "
            "Run with: CUDA_VISIBLE_DEVICES=0 python RL_traning.py --phase sft"
        )

    os.makedirs(SFT_DIR, exist_ok=True)
    tokenizer   = load_tokenizer(BASE_MODEL)
    bnb_config  = make_bnb_config()
    lora_config = make_lora_config()

    print("\nLoading student base model on GPU 0 …")
    print_gpu_memory("before SFT model load")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={'': 0},
    )
    print_gpu_memory("after SFT model load")

    def tokenize(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_PROMPT_LEN,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    train_ds = Dataset.from_list([{"text": build_sft_text(x)} for x in train_data])
    val_ds   = Dataset.from_list([{"text": build_sft_text(x)} for x in val_data])
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds   = val_ds.map(tokenize,   batched=True, remove_columns=["text"])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    args = TrainingArguments(
        output_dir=SFT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=2,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        dataloader_pin_memory=False,
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        peft_config=lora_config,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(SFT_DIR)
    tokenizer.save_pretrained(SFT_DIR)
    print(f"\nSFT model saved to {SFT_DIR}")

    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print_gpu_memory("after SFT cleanup")
    return SFT_DIR


# ============================================================
# PPO phase  —  no teacher in memory, uses cached rewards
# ============================================================
# Launch: CUDA_VISIBLE_DEVICES=0,1,2,3 python RL_traning.py --phase ppo
# Student  → GPU 0  (weights + gradients + activations)
# Ref      → GPU 1  (frozen, inference only)
# GPUs 2&3 → free (available for activation overflow)
# ============================================================

def load_value_head_model(model_path: str, tokenizer, device_id: int = 0):
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path,
        quantization_config=make_bnb_config(),
        device_map={'': device_id},   # single GPU — PPOTrainer requires all tensors on same device
        peft_config=make_lora_config(),
    )
    return model


def build_reference_model(sft_path: str, tokenizer, device_id: int = 1):
    ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        sft_path,
        quantization_config=make_bnb_config(),
        device_map={'': device_id},   # separate GPU from student
        peft_config=make_lora_config(),
    )
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    return ref


def run_ppo(
    train_data: List[Dict],
    val_data: List[Dict],
    sft_path: str,
    teacher_reward_cache: Dict[int, float],
):
    if not os.path.exists(sft_path):
        raise FileNotFoundError(
            f"SFT checkpoint not found at {sft_path}. "
            "Run --phase sft first."
        )

    os.makedirs(PPO_DIR, exist_ok=True)
    tokenizer = load_tokenizer(sft_path)

    print_gpu_memory("before PPO student load")
    student   = load_value_head_model(sft_path, tokenizer, device_id=0)
    ref_model = build_reference_model(sft_path, tokenizer, device_id=1)
    print_gpu_memory("after PPO student + ref load")

    ppo_config = PPOConfig(
        batch_size=PPO_BATCH_SIZE,
        mini_batch_size=PPO_MINI_BATCH_SIZE,
        learning_rate=PPO_LR,
        init_kl_coef=PPO_INIT_KL,
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=student,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    policy_device = next(
        (p.device for p in student.pretrained_model.parameters()),
        torch.device("cuda:0"),
    )
    print(f"Policy device: {policy_device}")

    # Unwrap once — bypasses ppo_trainer.generate() position_ids bug
    unwrapped = ppo_trainer.accelerator.unwrap_model(student)

    for epoch in range(PPO_EPOCHS):
        print(f"\n{'=' * 60}")
        print(f"PPO EPOCH {epoch + 1}/{PPO_EPOCHS}")
        print(f"{'=' * 60}")

        # Shuffle while keeping original indices for cache lookup
        indexed_data = list(enumerate(train_data))
        random.shuffle(indexed_data)

        batch_queries   = []
        batch_responses = []
        batch_rewards   = []

        for step, (orig_idx, ex) in enumerate(indexed_data):
            prompt = build_prompt(ex)

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_PROMPT_LEN,
                padding=False,
                return_attention_mask=True,
            )
            input_ids      = enc["input_ids"].to(policy_device)       # (1, seq_len)
            attention_mask = enc["attention_mask"].to(policy_device)  # (1, seq_len)

            with torch.no_grad():
                output_ids = unwrapped.pretrained_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    synced_gpus=False,
                )

            query_ids  = input_ids[0]                          # 1D for ppo_trainer.step
            new_tokens = output_ids[0][input_ids.shape[1]:]    # 1D response
            response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Combine live correctness reward + cached teacher reward
            correctness_reward   = compute_correctness_reward(response, str(ex["answer"]))
            cached_teacher_reward = teacher_reward_cache.get(orig_idx, 0.0)
            total_reward = max(-1.0, min(1.0,
                CORRECTNESS_REWARD_WEIGHT * correctness_reward
                + TEACHER_REWARD_WEIGHT   * cached_teacher_reward
            ))

            batch_queries.append(query_ids.to(policy_device))
            batch_responses.append(new_tokens.to(policy_device))
            batch_rewards.append(
                torch.tensor(total_reward, dtype=torch.float32, device=policy_device)
            )

            if (step + 1) % 20 == 0:
                print(
                    f"Step {step+1:4d} | total={total_reward: .3f} | "
                    f"correctness={correctness_reward: .3f} | "
                    f"teacher(cached)={cached_teacher_reward: .3f} | "
                    f"pred={response[:80]!r} | gold={str(ex['answer'])[:50]!r}"
                )

            if len(batch_queries) == PPO_BATCH_SIZE:
                torch.cuda.empty_cache()
                ppo_trainer.step(batch_queries, batch_responses, batch_rewards)
                batch_queries   = []
                batch_responses = []
                batch_rewards   = []

        # Flush remaining
        if len(batch_queries) == PPO_BATCH_SIZE:
            torch.cuda.empty_cache()
            ppo_trainer.step(batch_queries, batch_responses, batch_rewards)
        elif len(batch_queries) > 0:
            print(f"Skipping final incomplete batch of {len(batch_queries)} examples "
                f"(need {PPO_BATCH_SIZE}).")

        epoch_dir = os.path.join(PPO_DIR, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        student.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"Saved PPO checkpoint to {epoch_dir}")

        evaluate_model(
            student.pretrained_model,
            tokenizer,
            val_data,
            policy_device,
            f"PPO val epoch {epoch+1}",
        )

    student.save_pretrained(PPO_DIR)
    tokenizer.save_pretrained(PPO_DIR)
    print(f"\nPPO model saved to {PPO_DIR}")
    return PPO_DIR


# ============================================================
# Phase runners
# ============================================================

def phase_sft(train_data, val_data, test_data):
    """
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        python RL_traning.py --phase sft
    """
    print("\n" + "=" * 60)
    print("PHASE: SFT")
    print("=" * 60)

    sft_path = run_sft(train_data, val_data)

    sft_tokenizer = load_tokenizer(sft_path)
    sft_model     = AutoModelForCausalLM.from_pretrained(
        sft_path,
        device_map={'': 0},
        quantization_config=make_bnb_config(),
    )
    sft_device = next(sft_model.parameters()).device
    evaluate_model(sft_model, sft_tokenizer, val_data,  sft_device, "SFT val")
    evaluate_model(sft_model, sft_tokenizer, test_data, sft_device, "SFT test")

    free_gpu_memory(sft_model, sft_tokenizer)
    print_gpu_memory("after SFT evaluation cleanup")
    print(f"\nSFT complete. Checkpoint at: {sft_path}")
    print("\nNext — pre-compute teacher rewards:")
    print(
        "  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "python RL_traning.py --phase cache"
    )


def phase_cache(train_data):
    """
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        python RL_traning.py --phase cache
    """
    print("\n" + "=" * 60)
    print("PHASE: CACHE TEACHER REWARDS")
    print("=" * 60)

    teacher_tokenizer, teacher = load_teacher()
    diagnose_teacher(teacher_tokenizer, teacher)
    if train_data:
        sanity_check_teacher_reward(teacher_tokenizer, teacher, train_data[0])

    precompute_teacher_rewards(train_data, teacher_tokenizer, teacher)

    free_gpu_memory(teacher, teacher_tokenizer)
    print_gpu_memory("after teacher freed")
    print("\nCache complete.")
    print("\nNext — run PPO:")
    print(
        "  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "python RL_traning.py --phase ppo"
    )


def phase_ppo(train_data, val_data, test_data):
    """
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        python RL_traning.py --phase ppo
    """
    print("\n" + "=" * 60)
    print("PHASE: PPO")
    print("=" * 60)

    teacher_reward_cache = load_teacher_cache()
    print(f"Loaded {len(teacher_reward_cache)} cached teacher rewards.")

    ppo_path = run_ppo(
        train_data=train_data,
        val_data=val_data,
        sft_path=SFT_DIR,
        teacher_reward_cache=teacher_reward_cache,
    )

    # Final evaluation
    ppo_tokenizer = load_tokenizer(ppo_path)
    ppo_model     = AutoModelForCausalLM.from_pretrained(
        ppo_path,
        device_map={'': 0},
        quantization_config=make_bnb_config(),
    )
    ppo_device = next(ppo_model.parameters()).device
    evaluate_model(ppo_model, ppo_tokenizer, test_data, ppo_device, "PPO test")

    free_gpu_memory(ppo_model, ppo_tokenizer)
    print(f"\nPPO complete. Final checkpoint at: {ppo_path}")


def phase_cache_and_ppo(train_data, val_data, test_data):
    """
    CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        python RL_traning.py --phase cache_and_ppo
    """
    print("\n" + "=" * 60)
    print("PHASE: CACHE + PPO")
    print("=" * 60)

    # Cache teacher rewards first
    teacher_tokenizer, teacher = load_teacher()
    diagnose_teacher(teacher_tokenizer, teacher)
    if train_data:
        sanity_check_teacher_reward(teacher_tokenizer, teacher, train_data[0])

    teacher_reward_cache = precompute_teacher_rewards(
        train_data, teacher_tokenizer, teacher
    )

    # Free teacher — all 4 GPUs now available for PPO student
    print("\nFreeing teacher before PPO …")
    free_gpu_memory(teacher, teacher_tokenizer)
    print_gpu_memory("after teacher freed — GPUs available for PPO")

    # Run PPO with cached rewards
    ppo_path = run_ppo(
        train_data=train_data,
        val_data=val_data,
        sft_path=SFT_DIR,
        teacher_reward_cache=teacher_reward_cache,
    )

    # Final evaluation
    ppo_tokenizer = load_tokenizer(ppo_path)
    ppo_model     = AutoModelForCausalLM.from_pretrained(
        ppo_path,
        device_map={'': 0},
        quantization_config=make_bnb_config(),
    )
    ppo_device = next(ppo_model.parameters()).device
    evaluate_model(ppo_model, ppo_tokenizer, test_data, ppo_device, "PPO test")

    free_gpu_memory(ppo_model, ppo_tokenizer)
    print(f"\nAll done. Final checkpoint at: {ppo_path}")


# ============================================================
# Entry point
# ============================================================

def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parse_args()
    print_gpu_memory("at startup")
    print(f"\nRunning phase : {args.phase}")
    print(f"Visible GPUs  : {torch.cuda.device_count()}")

    train_data = load_json_dataset(TRAIN_JSON_PATH)
    val_data   = load_json_dataset(VAL_JSON_PATH)
    test_data  = load_json_dataset(TEST_JSON_PATH)
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    if args.phase == "sft":
        phase_sft(train_data, val_data, test_data)

    elif args.phase == "cache":
        phase_cache(train_data)

    elif args.phase == "ppo":
        phase_ppo(train_data, val_data, test_data)

    elif args.phase == "cache_and_ppo":
        phase_cache_and_ppo(train_data, val_data, test_data)


if __name__ == "__main__":
    main()
