import contextlib
import json
import sys, os
import time
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

from dataset_code.gen_questions.config import ExLlamaArguments
from transformers import HfArgumentParser
from datasets import load_dataset

import exllamav2
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache_Q4, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior software engineer reviewing code and its associated documentation.

You must read the code and documentation provided and generate a high-quality QA pair.
The best questions use the documentation as an important source of evidence, not just the code.

The output must be valid JSON with exactly THREE fields:
{
  "reasoning": "A brief explanation of why this question is grounded in both the code and documentation.",
  "question": "The realistic developer question.",
  "answer": "The accurate answer derived from the provided inputs."
}
"""

TASK_TEMPLATE = """
You are generating ONE realistic developer question for a code-assistant dataset.

Your task:
Read the implementation and the accompanying project text, then write exactly ONE issue-like question that a developer might genuinely raise during debugging, review, testing, integration, maintenance, or API adoption.

Implementation:
{code}

Project text:
{documentation}

Goal:
Generate a single self-contained question that requires combining details from both inputs.
The question should surface a non-obvious engineering concern, ambiguity, risk, edge case, hidden assumption, test gap, or likely follow-up action.

Hard requirements:
1. Output exactly one JSON object with exactly three keys: "reasoning", "question", "answer".
2. The question must be grounded in BOTH inputs together, not just one.
3. The question must not be a paraphrase, summary request, or basic explanation request.
4. The question must be specific, actionable, and useful to a real engineer.
5. Do not mention the words "code", "documentation", "docstring", "comment", or "issue".

Then output only:

{{
  "reasoning": "...",
  "question": "...",
  "answer": "..."
}}
"""

def build_prompt(code: str, documentation: str) -> str:
    user_content = TASK_TEMPLATE.format(code=code, documentation=documentation)
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT.strip()}"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content.strip()}"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_response(text: str) -> dict | None:
    text = text.strip()
    start = text.rfind("{")
    end = text.rfind("}")
    
    if start == -1 or end == -1 or end <= start:
        return None
        
    json_str = text[start:end+1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            cleaned = json_str.replace('\n', ' ')
            return json.loads(cleaned)
        except:
            return None

def save_data(data: list, data_file: str) -> None:
    if not data: return
    # Use absolute path to avoid confusion
    abs_path = os.path.abspath(data_file)
    with open(abs_path, "a") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

# ── Setup ──────────────────────────────────────────────────────────────────────

parser = HfArgumentParser(ExLlamaArguments)
model_args, = parser.parse_args_into_dataclasses()

# Load model and tokenizer
config = ExLlamaV2Config(model_args.model_dir)
config.arch_compat_overrides()
model = ExLlamaV2(config)

cache = ExLlamaV2Cache_Q4(model, max_seq_len=min(model_args.max_seq_len, 32768), lazy=True)
model.load_autosplit(cache, progress=True)

tokenizer = ExLlamaV2Tokenizer(config)

generator = ExLlamaV2DynamicGenerator(
    model=model,
    cache=cache,
    tokenizer=tokenizer,
    max_batch_size=16
)

gen_settings = ExLlamaV2Sampler.Settings(
    token_repetition_penalty=1.15,
    temperature=0.7,
    top_p=0.9
)

# Load and Filter Data
raw_data = load_dataset("json", data_files=model_args.dataset_path)["train"]
MAX_TOKENS = 24576
filtered_data = []

print(f" [*] Filtering dataset...")
for idx, s in enumerate(raw_data):
    if len(s["code"]) + len(s["text"]) > MAX_TOKENS * 3:
        ids = tokenizer.encode(build_prompt(s["code"], s["text"]))
        if ids.shape[-1] > MAX_TOKENS:
            continue
    s["_len"] = len(s["code"]) + len(s["text"])
    s["_idx"] = idx
    filtered_data.append(s)

filtered_data.sort(key=lambda x: x["_len"])
dataset_len = len(filtered_data)
print(f" [*] Ready with {dataset_len} samples.")

# ── Generation Loop ────────────────────────────────────────────────────────────

all_results = []
time_begin = time.time()
num_completions = 0
num_enqueued = 0
stop_id = tokenizer.single_id("<|eot_id|>")

with exllamav2.util.get_basic_progress() as progress:
    task = progress.add_task("[red]Generating", total=dataset_len)
    safety_margin = model_args.max_new_tokens + 128
    limit = (model_args.max_seq_len or 32768) - safety_margin

    while num_completions < dataset_len:
        
        # 1. Enqueue
        while num_enqueued < dataset_len and generator.num_remaining_jobs() < 500:
            chunk_size = 100
            end_idx = min(num_enqueued + chunk_size, dataset_len)
            jobs = []
            for idx in range(num_enqueued, end_idx):
                s = filtered_data[idx]
                ids = tokenizer.encode(build_prompt(s["code"], s["text"]))
                if ids.shape[-1] > limit: ids = ids[:, :limit]
                job = ExLlamaV2DynamicJob(
                    input_ids=ids, max_new_tokens=model_args.max_new_tokens,
                    gen_settings=gen_settings, stop_conditions=[stop_id],
                    decode_special_tokens=True, identifier=idx 
                )
                jobs.append(job)
            generator.enqueue(jobs)
            num_enqueued = end_idx
            print(f" [*] Enqueued up to sample {num_enqueued}")

        # 2. Iterate
        results = generator.iterate()
        
        for r in results:
            # Check for ANY variant of "finished"
            is_finished = (
                r.get("stage") == "FINISHED" or 
                r.get("stage") == 4 or 
                r.get("eos") == True
            )

            if is_finished:
                idx = r.get("identifier")
                text = r.get("full_completion", "")
                
                print(f" [SUCCESS] Sample {idx} complete. tokens: {len(text)//4}")
                
                parsed = parse_response(text)
                orig_sample = filtered_data[idx]
                output_sample = {k: v for k, v in orig_sample.items() if not k.startswith("_")}
                
                output_sample["reasoning"] = parsed.get("reasoning", "").strip() if parsed else ""
                output_sample["question"] = parsed.get("question", "").strip() if parsed else ""
                output_sample["answer"] = parsed.get("answer", "").strip() if parsed else ""
                output_sample["raw_completion"] = text
                
                all_results.append(output_sample)
                num_completions += 1
                progress.update(task, advance=1)
                
                # Save every single question immediately for real-time visibility
                if len(all_results) >= 1:
                    save_data(all_results, model_args.output_path)
                    all_results = []
        
        if not results:
            time.sleep(0.01)

if all_results:
    save_data(all_results, model_args.output_path)