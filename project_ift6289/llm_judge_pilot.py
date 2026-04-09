# print("Importing..")
import json
import random
from pathlib import Path
import re

print("Importing Exllama... ")
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = "/Tmp/mancasat/models/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"
DATASET_PATH = "final_aligned_dataset.jsonl"
OUTPUT_PATH = "pilot_judgments.jsonl"

SAMPLE_SIZE = 100 # how many examples we evaluate in the pilot 
BATCH_SIZE = 4 # how many prompts to send at once 
MAX_NEW_TOKENS = 120 # how many tokens is the model allowed to generate?

SYSTEM_PROMPT = """You are a strict judge for code-documentation alignment.
Return JSON only.
Schema:
{
  "label": "keep" | "borderline" | "reject",
  "reason": "short explanation"
}
"""

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def truncate_text(s, max_chars):
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[truncated]"

def build_prompt(row):
    kind = row.get("kind", "")
    text = row.get("text", "")
    code = row.get("code", "")

    text = truncate_text(text, 1200)
    code = truncate_text(code, 3000)

    user_prompt = f"""You are judging whether a documentation text is a good semantic match for the code.

Return JSON only:
{{
  "label": "keep" | "borderline" | "reject",
  "reason": "short explanation"
}}

Rules:
- keep: text clearly matches the code and is useful
- borderline: partial match, too generic, or needs more context
- reject: mismatched, stale, trivial, or unrelated
- Return ONLY valid JSON. Do not include any extra text.

DOCUMENTATION KIND:
{kind}

TEXT:
{text}

CODE:
{code}
"""

    # Llama-style chat formatting; adjust if your existing template differs
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

# for the model's json output - not necessarily always clean  

def safe_parse_json(raw_text):
    raw_text = (raw_text or "").strip()

    # Keep only text after the last assistant header if present
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    if marker in raw_text:
        raw_text = raw_text.split(marker)[-1].strip()

    # Find JSON objects that look like the target schema
    matches = re.findall(
        r'\{\s*"label"\s*:\s*"(keep|borderline|reject)"\s*,\s*"reason"\s*:\s*".*?"\s*\}',
        raw_text,
        flags=re.DOTALL
    )

    if matches:
        # re.findall only returns capture groups here, so use finditer instead
        pass

    for m in re.finditer(
        r'\{\s*"label"\s*:\s*"(keep|borderline|reject)"\s*,\s*"reason"\s*:\s*".*?"\s*\}',
        raw_text,
        flags=re.DOTALL
    ):
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            continue

    return {
        "label": "reject",
        "reason": f"Could not parse model output: {raw_text[:200]}"
    }

def main():
    # Load dataset
    rows = load_jsonl(DATASET_PATH)

    # Sample pilot set
    if len(rows) > SAMPLE_SIZE:
        sampled = random.sample(rows, SAMPLE_SIZE)
    else:
        sampled = rows

    # Load model
    print("[INFO] Loading model & tokenizer...")
    config = ExLlamaV2Config(MODEL_DIR)
    config.arch_compat_overrides()

    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len=8192, lazy=True)
    tokenizer = ExLlamaV2Tokenizer(config)

    model.load_autosplit(cache, progress=True)

    print("[INFO] Loading generator")
    generator = ExLlamaV2DynamicGenerator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
    )

    generator.warmup()

    prompts = [build_prompt(r) for r in sampled]

    results = []
    for i in range(0, len(prompts), BATCH_SIZE):
        print(f"[INFO] Prompts {i} to {i+BATCH_SIZE}...")
        batch_prompts = prompts[i:i+BATCH_SIZE]
        batch_rows = sampled[i:i+BATCH_SIZE]

        outputs = generator.generate(
            prompt=batch_prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            add_bos=False,
            stop_conditions=["<|eot_id|>"]
        )

        if isinstance(outputs, str):
            outputs = [outputs]

        for row, raw_output in zip(batch_rows, outputs):
            #print("raw output" , raw_output)
            parsed = safe_parse_json(raw_output)
            results.append({
                "pair_id": row.get("pair_id"),
                "repo_id": row.get("repo_id"),
                "file_path": row.get("file_path"),
                "pair_type": row.get("pair_type"),
                "heuristic_confidence": row.get("confidence"),
                "judge_output": parsed,
                "raw_model_output": raw_output
            })
            # print(f"{row.get('repo_id')} - {row.get('file_path')} is {parsed}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} pilot judgments to {OUTPUT_PATH}")

def count_outputs():
    keep_high_count = 0
    keep_medium_count = 0
    reject_high_count = 0
    reject_medium_count = 0
    high_count = 0
    medium_count = 0

    rows = load_jsonl(OUTPUT_PATH)

    for row in rows:
        #print(row)
        judgment = row["judge_output"]["label"]
        confidence = row["heuristic_confidence"]

        if judgment == "keep":
            if confidence == "high":
                keep_high_count += 1
                high_count += 1
            elif confidence == "medium":
                keep_medium_count += 1
                medium_count += 1
        elif judgment == "reject":
            if confidence == "high":
                high_count += 1
                reject_high_count += 1
            elif confidence == "medium":
                reject_medium_count += 1
                medium_count += 1

    print("high count: ", high_count)
    print("medium_count: ", medium_count)
    print(keep_high_count)
    print(keep_medium_count)
    print(reject_high_count)
    print(reject_medium_count)


if __name__ == "__main__":
    main()
    count_outputs()