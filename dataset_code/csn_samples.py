from datasets import load_dataset
import json

OUTPUT_FILE = "csn_samples.jsonl"
NUM_SAMPLES = 2000
LANGUAGES = ["python", "java"]

def fetch_csn_samples(lang, limit=2000):
    print(f"[INFO] Fetching {limit} samples for {lang} from CodeSearchNet (streaming)...")
    samples = []
    
    # Using streaming=True to avoid downloading the entire 1GB+ dataset
    ds = load_dataset("code_search_net", lang, split="train", streaming=True)
    
    count = 0
    for item in ds:
        # Extract relevant fields based on actual CodeSearchNet schema:
        # func_documentation_string, whole_func_string, func_path_in_repository
        samples.append({
            "uid": f"csn-{lang}-{count}",
            "repo_id": item.get("repository_name", "unknown"),
            "file_path": item.get("func_path_in_repository", "unknown"),
            "language": lang.capitalize(),
            "source": "docstring",
            "entity_type": "function",
            "confidence": "high",
            "text": item.get("func_documentation_string"),
            "code": item.get("whole_func_string")
        })
        count += 1
        if count % 100 == 0:
            print(f"  [PROGRESS] {count}/{limit}...")
        if count >= limit:
            break
            
    return samples

def main():
    all_samples = []
    
    for lang in LANGUAGES:
        try:
            samples = fetch_csn_samples(lang, NUM_SAMPLES)
            all_samples.extend(samples)
        except Exception as e:
            print(f"[ERR] Failed to fetch {lang}: {e}")
    
    print(f"[INFO] Collected {len(all_samples)} total samples from CodeSearchNet.")
    
    print(f"[SUCCESS] Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
