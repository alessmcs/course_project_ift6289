import json
import random
import re
import os

print("Importing Exllama... ")
# from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
# from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

# -----------------------------
# Config
# -----------------------------
MODEL_DIR = "/Tmp/mancasat/models/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"
DATASET_PATH = "training_dataset_final.jsonl"
OUTPUT_PATH = "pilot_judgments.jsonl"

MAX_NEW_TOKENS = 120

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
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[truncated]"

def build_prompt(row):
    source = row.get("source", "unknown")
    entity_type = row.get("entity_type", "unknown")
    language = row.get("language", "unknown")
    text = row.get("text", "")
    code = row.get("code", "")

    text = truncate_text(text, 1200)
    code = truncate_text(code, 3000)

    user_prompt = f"""You are judging if a documentation snippet is a GOOD semantic match for the provided code.

SOURCE TYPE: {source} ({entity_type})
LANGUAGE: {language}

DOCUMENTATION TEXT:
{text}

CODE/DIFF:
{code}

Return ONLY JSON:
{{
  "label": "keep" | "borderline" | "reject",
  "reason": "short explanation"
}}

Rules:
- "keep": The text describes the code logic accurately.
- "borderline": Too generic, slightly stale, or formatting issues.
- "reject": Totally unrelated, empty text, or completely wrong logic.
"""

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

def safe_parse_json(raw_text):
    raw_text = (raw_text or "").strip()

    marker = "<|start_header_id|>assistant<|end_header_id|>"
    if marker in raw_text:
        raw_text = raw_text.split(marker)[-1].strip()

    # Regex for a clean JSON block
    match = re.search(r'\{\s*"label"\s*:\s*"(keep|borderline|reject)"\s*,\s*"reason"\s*:\s*".*?"\s*\}', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return {
        "label": "reject",
        "reason": f"Could not parse model output: {raw_text[:100]}"
    }

def summarize_outputs(results):
    counts = {"keep": 0, "borderline": 0, "reject": 0}
    for row in results:
        label = row["judge_output"]["label"]
        if label in counts:
            counts[label] += 1

    print("\n[SUMMARY]")
    print(f"keep: {counts['keep']}")
    print(f"borderline: {counts['borderline']}")
    print(f"reject: {counts['reject']}")

def main():
    rows = load_jsonl(DATASET_PATH)
    sampled = rows # Process EVERYTHING
    print(f"[INFO] Starting full evaluation of {len(sampled)} samples...")

    print("[INFO] Loading model & tokenizer...")
    config = ExLlamaV2Config(MODEL_DIR)
    config.arch_compat_overrides()

    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len=8192, lazy=True)
    tokenizer = ExLlamaV2Tokenizer(config)

    model.load_autosplit(cache, progress=True)

    print("[INFO] Loading generator...")
    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.0 # Deterministic
    settings.top_k = 1
    settings.top_p = 1.0
    settings.token_repetition_penalty = 1.0

    results = []

    # Open output file in append mode to avoid losing data if it crashes
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, row in enumerate(sampled, start=1):
            if i % 10 == 0:
                print(f"[INFO] Prompt {i}/{len(sampled)}...")
                
            prompt = build_prompt(row)

            raw_output = generator.generate_simple(
                prompt,
                settings,
                MAX_NEW_TOKENS,
                add_bos=False,
                stop_token=tokenizer.single_id("<|eot_id|>")
            )

            parsed = safe_parse_json(raw_output)

            result_entry = {
                "uid": row.get("uid"),
                "repo_id": row.get("repo_id"),
                "language": row.get("language"),
                "source": row.get("source"),
                "entity_type": row.get("entity_type"),
                "heuristic_confidence": row.get("confidence"),
                "text": row.get("text"),
                "code": row.get("code"),
                "judge_output": parsed,
                "raw_model_output": raw_output
            }
            results.append(result_entry)
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\nSaved {len(results)} judgments to {OUTPUT_PATH}")
    summarize_outputs(results)
    filter_and_save()

def filter_and_save():
    final_output = "training_dataset_verified.jsonl"
    print(f"[INFO] Filtering out rejected entries and saving to {final_output}...")
    
    count = 0
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f_in, \
         open(final_output, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip(): continue
            item = json.loads(line)
            
            # If text/code are missing (older version), try to recover from raw_model_output
            if "text" not in item and "raw_model_output" in item:
                raw = item["raw_model_output"]
                t_match = re.search(r"DOCUMENTATION TEXT:\n(.*?)\n\nCODE/DIFF:", raw, re.DOTALL)
                c_match = re.search(r"CODE/DIFF:\n(.*?)\n\nReturn ONLY JSON:", raw, re.DOTALL)
                if t_match: item["text"] = t_match.group(1).strip()
                if c_match: item["code"] = c_match.group(1).strip()

            if item["judge_output"]["label"] in ["keep", "borderline"]:
                # Save just the original training keys to keep it clean
                clean_item = {
                    "uid": item.get("uid"),
                    "repo_id": item.get("repo_id"),
                    "language": item.get("language"),
                    "source": item.get("source"),
                    "entity_type": item.get("entity_type"),
                    "text": item.get("text", ""),
                    "code": item.get("code", "")
                }
                f_out.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
                count += 1
                
    print(f"[SUCCESS] Final verified dataset: {count} samples.")



if __name__ == "__main__":
    # main()
    filter_and_save()