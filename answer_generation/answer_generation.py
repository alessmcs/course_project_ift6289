import contextlib
import json
import sys, os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import HfArgumentParser
from datasets import load_dataset

import exllamav2
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler

# ── Config ─────────────────────────────────────────────────────────────────────

SMALL_MODEL_DIR  = "/data/rech/jaouaime/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
BIG_MODEL_DIR    = "/data/rech/jaouaime/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"

SMALL_MODEL_GPUS = [0]        # 8B  on GPU 0
BIG_MODEL_GPUS   = [1, 2, 3]  # 70B on GPUs 1, 2, 3

DATASET_PATH     = "results.jsonl"   # output of question generation
OUTPUT_PATH      = "sft_dataset.jsonl"
SAVE_STEPS       = 50

MAX_SEQ_LEN_SMALL = 8192
MAX_SEQ_LEN_BIG   = 8192
MAX_NEW_TOKENS_SMALL = 128   # raw answer from 8B
MAX_NEW_TOKENS_BIG   = 160   # improved answer from 70B

# ── Prompts ────────────────────────────────────────────────────────────────────

ANSWER_SYSTEM = """You are a software engineer. Answer the question based on the code and documentation provided.
Be technical, precise, and concise. Do not restate the question."""

ANSWER_TEMPLATE = """Code:
{code}

Documentation:
{documentation}

Question:
{question}

Answer directly and technically."""

IMPROVE_SYSTEM = """You are a senior software engineer doing a factual review of a draft answer about code.

Your process must be:
1. Read the code carefully line by line.
2. Identify every claim in the draft answer.
3. Check each claim directly against the code — if the code does not support it, remove or correct it.
4. Write the final answer using only what the code actually does.

Hard rules:
- Every statement you write must be traceable to a specific line in the code.
- Do not generalize or infer behavior the code does not implement.
- Do not preserve false claims just because they sound plausible.
- Write only the final answer — no preamble, no commentary, no explanation of your changes."""


IMPROVE_TEMPLATE = """Code:
{code}

Documentation:
{documentation}

Question:
{question}

Draft answer (may contain errors):
{draft_answer}

First, verify each claim in the draft against the code above.
Then write the corrected, complete answer grounded strictly in what the code does.
Do not include anything the code does not explicitly implement."""
# ── Llama-3 chat template ──────────────────────────────────────────────────────

def build_prompt(system: str, user: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system.strip()}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user.strip()}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

def build_answer_prompt(code: str, documentation: str, question: str) -> str:
    return build_prompt(
        system=ANSWER_SYSTEM,
        user=ANSWER_TEMPLATE.format(
            code=code,
            documentation=documentation,
            question=question,
        )
    )

def build_improve_prompt(code: str, documentation: str, question: str, draft_answer: str) -> str:
    return build_prompt(
        system=IMPROVE_SYSTEM,
        user=IMPROVE_TEMPLATE.format(
            code=code,
            documentation=documentation,
            question=question,
            draft_answer=draft_answer,
        )
    )

# ── Helpers ────────────────────────────────────────────────────────────────────

def generate_single(generator, tokenizer, prompt: str, max_new_tokens: int, gen_settings) -> str:
    """Run a single synchronous generation — enqueue, drain, return text."""
    input_ids = tokenizer.encode(prompt, encode_special_tokens=True, add_bos=False)
    job = ExLlamaV2DynamicJob(
        input_ids=input_ids,
        gen_settings=gen_settings,
        max_new_tokens=max_new_tokens,
        stop_conditions=[tokenizer.single_id("<|eot_id|>")],
        token_healing=True,
        identifier=0,
    )
    generator.enqueue(job)
    output = ""
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if result["eos"]:
                output = result["full_completion"].strip()
    return output

def is_valid(text: str, min_words: int = 8) -> bool:
    return bool(text) and len(text.strip().split()) >= min_words

def save_data(data: list, path: str) -> None:
    with open(path, "a") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

# ── Load models ────────────────────────────────────────────────────────────────

def load_model(model_dir: str, gpu_ids: list, max_seq_len: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    config = ExLlamaV2Config(model_dir)
    config.arch_compat_overrides()
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len=max_seq_len, lazy=True)
    model.load_autosplit(cache, progress=True)
    tokenizer = ExLlamaV2Tokenizer(config)
    generator = ExLlamaV2DynamicGenerator(model=model, cache=cache, tokenizer=tokenizer)
    return generator, tokenizer

print("Loading 8B model (GPU 0)...")
# ExLlamaV2 uses CUDA_VISIBLE_DEVICES at config time, so we manage device assignment
# via split_mode / device_map instead — load both before setting env vars

small_config = ExLlamaV2Config(SMALL_MODEL_DIR)
small_config.arch_compat_overrides()
small_model = ExLlamaV2(small_config)
small_cache = ExLlamaV2Cache(small_model, max_seq_len=MAX_SEQ_LEN_SMALL, lazy=True)
small_model.load_autosplit(small_cache, progress=True)
small_tokenizer = ExLlamaV2Tokenizer(small_config)
small_generator = ExLlamaV2DynamicGenerator(
    model=small_model, cache=small_cache, tokenizer=small_tokenizer
)

small_settings = ExLlamaV2Sampler.Settings(
    token_repetition_penalty=1.1,
    temperature=0.2,
)

print("Loading 70B model (GPUs 1, 2, 3)...")
big_config = ExLlamaV2Config(BIG_MODEL_DIR)
big_config.arch_compat_overrides()
big_model = ExLlamaV2(big_config)
big_cache = ExLlamaV2Cache(big_model, max_seq_len=MAX_SEQ_LEN_BIG, lazy=True)
big_model.load_autosplit(big_cache, progress=True)
big_tokenizer = ExLlamaV2Tokenizer(big_config)
big_generator = ExLlamaV2DynamicGenerator(
    model=big_model, cache=big_cache, tokenizer=big_tokenizer
)

big_settings = ExLlamaV2Sampler.Settings(
    token_repetition_penalty=1.1,
    temperature=0.1,  # lower temp for the improver — we want precision
)

# ── Dataset ────────────────────────────────────────────────────────────────────

data = load_dataset("json", data_files=DATASET_PATH)["train"]
data = data.filter(lambda x: x["question"] and x["question"].strip())
print(f"Dataset size: {len(data)}")

# ── Main loop ──────────────────────────────────────────────────────────────────

samples   = []
skipped   = 0
time_begin = time.time()

with exllamav2.util.get_basic_progress() as progress:
    task = progress.add_task("[red]Processing", total=len(data), name="Samples")

    for idx, sample in enumerate(data):
        code          = sample["code"].strip()
        documentation = sample["documentation"].strip()
        question      = sample["question"].strip()

        # ── Step 1: 8B generates raw answer ───────────────────────────────────
        answer_prompt = build_answer_prompt(code, documentation, question)
        raw_answer = generate_single(
            small_generator, small_tokenizer,
            answer_prompt, MAX_NEW_TOKENS_SMALL, small_settings
        )

        if not is_valid(raw_answer):
            print(f"[SKIP] {idx}: 8B answer too short.")
            skipped += 1
            progress.update(task, advance=1)
            continue

        # ── Step 2: 70B improves the raw answer ───────────────────────────────
        improve_prompt = build_improve_prompt(code, documentation, question, raw_answer)
        improved_answer = generate_single(
            big_generator, big_tokenizer,
            improve_prompt, MAX_NEW_TOKENS_BIG, big_settings
        )

        if not is_valid(improved_answer):
            print(f"[SKIP] {idx}: 70B improved answer too short, using raw.")
            improved_answer = raw_answer  # fallback to raw if improver fails

        # ── Stats ──────────────────────────────────────────────────────────────
        elapsed = time.time() - time_begin
        rpm = (idx + 1) / (elapsed / 60)
        print(f"\n--- {idx} | rpm {rpm:.1f} ---")
        print(f"Q:  {question[:80]}...")
        print(f"8B: {raw_answer[:80]}...")
        print(f"70B:{improved_answer[:80]}...")

        samples.append(dict(
            task_id=sample["task_id"],
            code=code,
            documentation=documentation,
            question=question,
            raw_answer=raw_answer,        # kept for debugging / DPO rejected side
            improved_answer=improved_answer,  # SFT training target
        ))

        # ── Incremental save ───────────────────────────────────────────────────
        if len(samples) % SAVE_STEPS == 0:
            save_data(samples[-SAVE_STEPS:], OUTPUT_PATH)
            print(f"[SAVE] {len(samples)} samples saved so far.")

        progress.update(task, advance=1)

# ── Final save ─────────────────────────────────────────────────────────────────
save_data(samples, OUTPUT_PATH)
print(f"\nDone. {len(samples)} samples saved, {skipped} skipped.")
print(f"Output: {OUTPUT_PATH}")