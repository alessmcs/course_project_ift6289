# Performs alignment between source code and documentation (docstrings, javadocs, commit messages, inline comments)
# Steps: 
# 1. Filter for repos whose score is >= 0.55
# 2. Build a code index with ASTs, (file path, start/end lines, full code text)
# 3. Perform strong alignment for docstrings and javadocs
# 4. Perform medium alignment for inline comments 
# 5. Align commit messages to files (Keep: new_start -> new_start + new_len)

import json
import os
import re
import io
import zipfile
import ast
import requests
import random 
from pathlib import Path
import uuid
from dotenv import load_dotenv

# Reusing some logic from file_metrics/eval_repo_quality if needed
# but keeping this script self-contained for simplicity and robustness.

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
}

REPO_SCORES_FILE = "repo_scores.json"
FILE_ENTITIES_FILE = "file_entities.jsonl"
COMMIT_ENTITIES_FILE = "commit_entities.jsonl"
ACCEPTED_FILES_FILE = "accepted_repo_files.jsonl"
ACCEPTED_REPOS_FILE = "accepted_repos.jsonl"

FINAL_OUTPUT_FILE = "final_aligned_dataset.jsonl"

CORE_EXTENSIONS = {".py", ".java"}

# Tree-sitter setup for Java
try:
    from tree_sitter_language_pack import get_parser
    JAVA_PARSER = get_parser("java")
except ImportError:
    import tree_sitter_java
    from tree_sitter import Language, Parser
    JAVA_LANG = Language(tree_sitter_java.language(), "java")
    JAVA_PARSER = Parser()
    JAVA_PARSER.set_language(JAVA_LANG)

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------
# AST Parsing Helpers
# -------------------------

def extract_python_nodes(source_text):
    nodes = []
    try:
        tree = ast.parse(source_text)
        lines = source_text.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = getattr(node, "lineno", 1)
                end_line = getattr(node, "end_lineno", start_line)
                nodes.append({
                    "name": getattr(node, "name", "unknown"),
                    "type": "function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class",
                    "start": start_line,
                    "end": end_line,
                    "code": "\n".join(lines[start_line-1:end_line])
                })
    except SyntaxError:
        pass
    return nodes

def extract_java_nodes(source_text):
    nodes = []
    source_bytes = source_text.encode("utf-8")
    tree = JAVA_PARSER.parse(source_bytes)
    lines = source_text.splitlines()

    def walk(node):
        if node.type in {"method_declaration", "constructor_declaration", "class_declaration", "interface_declaration"}:
            name_node = node.child_by_field_name("name")
            name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8") if name_node else "unknown"
            
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            nodes.append({
                "name": name,
                "type": "function" if "method" in node.type or "constructor" in node.type else "class",
                "start": start_line,
                "end": end_line,
                "code": "\n".join(lines[start_line-1:end_line])
            })
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return nodes

# -------------------------
# Patch Parsing
# -------------------------

def parse_patch_ranges(patch_text):
    """Extracts (new_start, new_end) ranges from a git patch."""
    ranges = []
    if not patch_text:
        return ranges
    
    # Hunk header: @@ -old_start,old_len +new_start,new_len @@
    hunk_headers = re.findall(r"@@ -\d+,\d+ \+(\d+),(\d+) @@", patch_text)
    for start, length in hunk_headers:
        start_line = int(start)
        end_line = start_line + int(length) - 1
        ranges.append((start_line, end_line))
    return ranges

# -------------------------
# Main Logic
# -------------------------

def main():
    print("[INFO] Starting alignment pipeline...")

    # 1. Filter Repos
    # scores = load_json(REPO_SCORES_FILE)
    # filtered_repos = [s for s in scores if s["scores"]["final_score"] >= 0.55]

    filtered_repos = load_jsonl(ACCEPTED_REPOS_FILE) # for now, until you fix the rate limit problem on eval_repo_quality.py

    # take a sample of 15 repos while you debug ; also not all accepted_repos have been parsed so only do first 100
    random.seed(42) # for reproducibility
    #filtered_repos = random.sample(filtered_repos[:100], 15)
    filtered_repo_ids = {s["repo_id"] for s in filtered_repos}
    repo_meta_map = {s["repo_id"]: s for s in filtered_repos}
    
    # print(f"[INFO] Filtered {len(filtered_repo_ids)} repos (score >= 0.55)")

    # Load entities
    all_file_entities = load_jsonl(FILE_ENTITIES_FILE)
    all_commit_entities = load_jsonl(COMMIT_ENTITIES_FILE)
    
    # Map entities to repos
    repo_file_entities = {}
    for ent in all_file_entities:
        if ent["repo_id"] in filtered_repo_ids:
            repo_file_entities.setdefault(ent["repo_id"], []).append(ent)
            
    repo_commit_entities = {}
    for ent in all_commit_entities:
        if ent["repo_id"] in filtered_repo_ids:
            repo_commit_entities.setdefault(ent["repo_id"], []).append(ent)

    final_pairs = []

    # 2. Iterate through repos
    for repo_id in filtered_repo_ids:
        repo_row = repo_meta_map[repo_id]
        commit_sha = repo_row.get("commit_sha")
        if not commit_sha: continue

        print(f"[PROCESS] Alignment for {repo_id}...")

        # Download zipball for source code access
        zip_url = f"https://api.github.com/repos/{repo_id}/zipball/{commit_sha}"
        try:
            resp = requests.get(zip_url, headers=HEADERS, timeout=60)
            if resp.status_code != 200:
                print(f"  [ERR] Failed to download zip: {resp.status_code}")
                continue
            
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                namelist = z.namelist()
                prefix = namelist[0].split("/")[0] + "/"
                
                # Build Code Index for this repo
                code_index = {} # path -> [nodes]
                
                # We only process files that are in our file_entities or commit_entities
                # Also only focus on core files (.py)
                needed_files = set()
                for ent in repo_file_entities.get(repo_id, []):
                    if ent["file_path"].endswith(".py") or ent["file_path"].endswith(".java"):
                        needed_files.add(ent["file_path"])
                for ent in repo_commit_entities.get(repo_id, []):
                    if ent["file_path"].endswith(".py") or ent["file_path"].endswith(".java"):
                        needed_files.add(ent["file_path"])

                # Also need any file mentioned in accepted_repo_files if we want full coverage
                # but let's stick to needed_files for efficiency.

                source_cache = {}

                for zip_path in namelist:
                    if zip_path.startswith(prefix) and not zip_path.endswith("/"):
                        repo_path = zip_path[len(prefix):]
                        if repo_path in needed_files:
                            try:
                                with z.open(zip_path) as f:
                                    content = f.read().decode("utf-8", errors="replace")
                                    source_cache[repo_path] = content
                                    
                                    if repo_path.endswith(".py"):
                                        code_index[repo_path] = extract_python_nodes(content)
                                    elif repo_path.endswith(".java"):
                                        code_index[repo_path] = extract_java_nodes(content)
                            except Exception as e:
                                print(f"  [WARN] Failed to parse {repo_path}: {e}")

                # 3 & 4. Strong + Medium Alignment (File Entities)
                for ent in repo_file_entities.get(repo_id, []):
                    path = ent["file_path"]
                    if path not in code_index: continue
                    
                    e_start = ent.get("entity_start_line")
                    e_end = ent.get("entity_end_line")
                    if e_start is None: continue
                    
                    # Find smallest enclosing node
                    nodes = code_index[path]
                    best_node = None
                    min_size = float('inf')
                    
                    for node in nodes:
                        if node["start"] <= e_start and node["end"] >= e_end:
                            size = node["end"] - node["start"]
                            if size < min_size:
                                min_size = size
                                best_node = node
                    
                    if best_node:
                        source_type = ent.get("source_type") or ent.get("doc_type")
                        text = ent["doc_text"]
                        
                        # Filtering
                        if len(re.findall(r"\b\w+\b", text)) < 5 and source_type in {"docstring", "javadoc"}:
                            continue
                        if best_node["code"].count("\n") < 2: # Min 3 lines
                            continue
                            
                        final_pairs.append({
                            "uid": str(uuid.uuid4()),
                            "repo_id": repo_id,
                            "file_path": path,
                            "source": source_type,
                            "confidence": ent.get("alignment_confidence", "medium"),
                            "text": text,
                            "code": best_node["code"]
                        })

                # 5. Commit -> Code Alignment
                for ent in repo_commit_entities.get(repo_id, []):
                    path = ent["file_path"]
                    if path not in code_index: continue
                    
                    patch = ent.get("patch_text")
                    if not patch: continue
                    
                    message = ent["doc_text"]
                    if not message or len(re.findall(r"\b\w+\b", message)) < 5:
                        continue
                        
                    changed_ranges = parse_patch_ranges(patch)
                    touched_nodes = set()
                    
                    nodes = code_index[path]
                    for r_start, r_end in changed_ranges:
                        for node in nodes:
                            # If hunk overlaps with node
                            if not (r_end < node["start"] or r_start > node["end"]):
                                touched_nodes.add(node["name"]) # Using name as key for simplicity
                    
                    if touched_nodes:
                        conf = "high" if len(touched_nodes) == 1 else "medium"
                        final_pairs.append({
                            "uid": str(uuid.uuid4()),
                            "repo_id": repo_id,
                            "file_path": path,
                            "source": "commit_message",
                            "confidence": conf,
                            "text": message,
                            "code": patch
                        })

        except Exception as e:
            print(f"  [FATAL] Error processing repo {repo_id}: {e}")

    # 7 & 8. Deduplication + Cleaning
    print(f"[INFO] Cleaning and deduplicating {len(final_pairs)} pairs...")
    
    unique_pairs = {}
    for p in final_pairs:
        # Key by code + text to avoid exact duplicates
        key = (p["code"].strip(), p["text"].strip())
        if key not in unique_pairs:
            unique_pairs[key] = p
        else:
            # Keep the one with higher confidence if duplicate
            if p["confidence"] == "high" and unique_pairs[key]["confidence"] != "high":
                unique_pairs[key] = p

    clean_data = list(unique_pairs.values())
    
    # 9. Save Dataset
    print(f"[SUCCESS] Saving {len(clean_data)} aligned pairs to {FINAL_OUTPUT_FILE}")
    with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in clean_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # clear output file
    with open(FINAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass
    main()
