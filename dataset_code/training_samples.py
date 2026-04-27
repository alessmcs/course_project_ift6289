import json
import random
import os
from collections import defaultdict

INPUT_FILE = "training_dataset_verified.jsonl"
CSN_FILE = "csn_samples.jsonl"
OUTPUT_FILE = "training_dataset_final.jsonl"

QUOTAS = {
    "docstring-function": 700,
    "docstring-class": 700,
    "comment-function": 2000,
    "comment-class": 2000,
    "commit-message": 2000,
    "csn": 700
}

ALLOWED_LANGUAGES = ["Python", "Java"]

def get_category(item):
    source = item.get("source")
    etype = item.get("entity_type")
    
    if source in ["docstring", "javadoc"]:
        if etype == "function": return "docstring-function"
        if etype == "class": return "docstring-class"
    
    if source == "inline_comment":
        if etype == "function": return "comment-function"
        if etype == "class": return "comment-class"
        
    if source == "commit_message":
        return "commit-message"
        
    return None

def main():
    print(f"[INFO] Reading {INPUT_FILE}...")
    
    # Buckets: (category, language) -> [items]
    buckets = defaultdict(list)
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            cat = get_category(item)
            lang = item.get("language", "unknown")
            
            if cat and lang in ALLOWED_LANGUAGES:
                buckets[(cat, lang)].append(item)
                
    all_categories = sorted(list(set(k[0] for k in buckets.keys())))
    all_languages = sorted(list(set(k[1] for k in buckets.keys())))
    
    print(f"[INFO] Found categories: {all_categories}")
    print(f"[INFO] Found languages: {all_languages}")
    
    final_samples = []
    
    for (cat, lang) in sorted(buckets.keys()):
        quota = QUOTAS.get(cat, 0)
        if quota == 0: continue
        
        # Balance by language within each category quota
        langs_for_this_cat = [l for (c, l) in buckets.keys() if c == cat]
        per_lang = quota // len(langs_for_this_cat)
        
        items = buckets[(cat, lang)]
        count = min(len(items), per_lang)
        final_samples.extend(random.sample(items, count))
        print(f"  - {cat} ({lang}): {count}/{len(items)}")
        
    # Add CSN samples
    if os.path.exists(CSN_FILE):
        print(f"[INFO] Sampling {QUOTAS['csn']} from {CSN_FILE}...")
        csn_items = []
        with open(CSN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): csn_items.append(json.loads(line))
        
        csn_count = min(len(csn_items), QUOTAS['csn'])
        final_samples.extend(random.sample(csn_items, csn_count))
        print(f"  - CodeSearchNet: {csn_count}/{len(csn_items)}")

    print(f"[INFO] Total samples collected: {len(final_samples)}")
    
    random.shuffle(final_samples)
    
    print(f"[SUCCESS] Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in final_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
