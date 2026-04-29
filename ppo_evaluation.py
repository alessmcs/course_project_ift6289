# evaluate_final.py  — run this after PPO finishes
import gc, os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from RL_traning import (
    load_tokenizer, make_bnb_config, load_json_dataset,
    build_prompt, generate_answer, answers_match,
    leakage_penalty, PPO_DIR, OUTPUT_DIR,
    TEST_JSON_PATH, VAL_JSON_PATH,
)

test_data = load_json_dataset(TEST_JSON_PATH)
val_data  = load_json_dataset(VAL_JSON_PATH)

tokenizer = load_tokenizer(PPO_DIR)
model     = AutoModelForCausalLM.from_pretrained(
    PPO_DIR,
    device_map={'': 0},
    quantization_config=make_bnb_config(),
)
device = next(model.parameters()).device

results_dir = os.path.join(OUTPUT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

for split_name, split_data in [("val", val_data), ("test", test_data)]:
    total, exact, leaked, records = len(split_data), 0, 0, []

    for i, ex in enumerate(split_data):
        prompt  = build_prompt(ex)
        pred    = generate_answer(model, tokenizer, prompt, device)
        correct = answers_match(pred, str(ex["answer"]))
        leak    = leakage_penalty(pred) > 0

        if correct: exact  += 1
        if leak:    leaked += 1

        records.append({
            "index":         i,
            # ── full input context ──────────────────────────
            "code":          ex["code"],
            "documentation": ex["documentation"],
            "question":      ex["question"],
            "prompt":        prompt,
            # ── outputs ─────────────────────────────────────
            "gold":          str(ex["answer"]),
            "pred":          pred,
            # ── diagnostics ─────────────────────────────────
            "correct":       correct,
            "leaked":        leak,
        })

        if (i + 1) % 100 == 0:
            print(f"[{split_name}] {i+1}/{total}  acc so far: {exact/(i+1):.3f}")

    accuracy  = exact  / total
    leak_rate = leaked / total

    out = {
        "split":     split_name,
        "accuracy":  accuracy,
        "leak_rate": leak_rate,
        "total":     total,
        "results":   records,
    }

    path = os.path.join(results_dir, f"ppo_{split_name}_predictions.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nPPO {split_name} accuracy:  {accuracy:.3f}")
    print(f"PPO {split_name} leak rate: {leak_rate:.3f}")
    print(f"Saved to {path}")