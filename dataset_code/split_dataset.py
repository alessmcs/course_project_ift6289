import json
import random
import os
from collections import defaultdict

def split_dataset(input_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits the dataset into train, validation, and test sets.
    Ensures that all samples from the same repository stay together to avoid data leakage.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Group entries by repo_id
    repo_to_items = defaultdict(list)
    total_items = 0
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            repo_id = item.get('repo_id', 'unknown')
            repo_to_items[repo_id].append(item)
            total_items += 1

    repos = list(repo_to_items.keys())
    print(f"Found {total_items} items across {len(repos)} repositories.")

    # Shuffle repos for a random split
    random.seed(seed)
    random.shuffle(repos)

    # Calculate split indices
    n_repos = len(repos)
    train_split = int(n_repos * train_ratio)
    val_split = train_split + int(n_repos * val_ratio)

    train_repos = repos[:train_split]
    val_repos = repos[train_split:val_split]
    test_repos = repos[val_split:]

    def save_split(filename, repo_list):
        count = 0
        with open(filename, 'w', encoding='utf-8') as f:
            for repo_id in repo_list:
                for item in repo_to_items[repo_id]:
                    f.write(json.dumps(item) + '\n')
                    count += 1
        return count

    print("\nSaving splits...")
    train_count = save_split('train.jsonl', train_repos)
    val_count = save_split('val.jsonl', val_repos)
    test_count = save_split('test.jsonl', test_repos)

    print(f"Train: {train_count} items ({len(train_repos)} repos) - {train_count/total_items:.1%}")
    print(f"Val:   {val_count} items ({len(val_repos)} repos) - {val_count/total_items:.1%}")
    print(f"Test:  {test_count} items ({len(test_repos)} repos) - {test_count/total_items:.1%}")
    print("\nSplit complete!")

if __name__ == "__main__":
    split_dataset('results.jsonl')
