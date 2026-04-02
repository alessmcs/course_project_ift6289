import os
import json
import random
import requests
from dotenv import load_dotenv

candidates_file = "candidate_repos.json"
accepted_repos_file = "accepted_repos.jsonl"

load_dotenv() #?? 
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN is not set")

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
}

ALLOWED_LICENSES = {
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Apache-2.0",
    "GPL-2.0",
    "GPL-3.0",
    "LGPL-2.1",
    "LGPL-3.0",
    "AGPL-3.0",
    "ISC",
    "CC0-1.0",
    "Unlicense",
    "MPL-2.0",
    "EPL-1.0",
    "EPL-2.0",
    "Zlib",
    "Python-2.0",
    "0BSD",
    "NOASSERTION",
    "OTHER",
    None # Allow if no license is detected (often means it was not in a standard file)
}

EXCLUDED_DIR_NAMES = {
    "test", "tests", "testing", "unittests",
    "example", "examples", "samples", "sample",
    "docs", "doc",
    "generated", "gen",
    "build", "bin", "dist",
    "target",
    "venv", ".venv", "env", ".env",
    "__pycache__", ".ipynb_checkpoints",
    ".git", ".github", ".circleci", ".gitlab",
    "node_modules", "bower_components"
}

def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

def normalize_github_url(url):
    if "github.com/" not in url:
        return None

    tail = url.split("github.com/")[1]
    tail = tail.split("?")[0].split("#")[0].strip("/")
    parts = tail.split("/")

    if len(parts) < 2:
        return None

    owner, repo = parts[0], parts[1]
    return f"https://github.com/{owner}/{repo}"

def get_repo_metadata(gh_link):
    gh_link = normalize_github_url(gh_link)
    if gh_link is None:
        return None

    owner_repo = gh_link.split("github.com/")[1]
    api_url = f"https://api.github.com/repos/{owner_repo}"

    response = requests.get(api_url, headers=HEADERS, timeout=10)

    if response.status_code == 200:
        return response.json()

    print(f"[ERROR] metadata fetch failed for {gh_link}: {response.status_code}")
    print(response.text[:300])
    return None

def is_allowed_license(repo_metadata):
    lic = repo_metadata.get("license")
    if not lic:
        return None in ALLOWED_LICENSES, None

    spdx = lic.get("spdx_id")
    if spdx is None:
        return None in ALLOWED_LICENSES, None

    # Compare case-insensitively (SPDX IDs in ALLOWED_LICENSES are uppercase)
    is_allowed = spdx.upper() in {str(l).upper() for l in ALLOWED_LICENSES if l is not None}
    
    # Also handle the None case explicitly if spdx is somehow a string "None"
    if not is_allowed and spdx.upper() == "NONE":
         is_allowed = None in ALLOWED_LICENSES

    return is_allowed, spdx

def get_repo_contents(repo_metadata):
    gh_link = repo_metadata.get("repo_url") or repo_metadata.get("github_link")
    commit_sha = repo_metadata.get("commit_sha")
    print(f"[INFO] Fetching full tree for {gh_link}")
    
    gh_link_norm = normalize_github_url(gh_link)
    if gh_link_norm is None:
        return None

    owner_repo = gh_link_norm.split("github.com/")[1]
    # Use Git Trees API to get all files recursively in one call
    tree_url = f"https://api.github.com/repos/{owner_repo}/git/trees/{commit_sha}?recursive=1"
    
    response = requests.get(tree_url, headers=HEADERS, timeout=20)

    if response.status_code != 200:
        print(f"[ERROR] tree fetch failed for {owner_repo}: {response.status_code}")
        return None

    tree_data = response.json()
    if tree_data.get("truncated"):
        print(f"[WARN] Tree for {owner_repo} is truncated!")

    all_items = tree_data.get("tree", [])
    
    # Check if 'src' directory exists
    has_src = any(item["path"] == "src" and item["type"] == "tree" for item in all_items)
    
    collected_files = []
    for item in all_items:
        if item["type"] != "blob": # 'blob' is a file
            continue
            
        path = item["path"]
        path_parts = path.lower().split("/")
        
        # Skip excluded directories
        if any(d in EXCLUDED_DIR_NAMES for d in path_parts):
            continue
            
        if has_src:
            # If 'src' exists, we only want files inside it
            if path.startswith("src/"):
                # Normalize format to match previous 'item' structure
                collected_files.append({
                    "path": path,
                    "sha": item["sha"],
                    "type": "file"
                })
        else:
            # Otherwise, collect everything from root
            collected_files.append({
                "path": path,
                "sha": item["sha"],
                "type": "file"
            })

    return collected_files


def get_latest_commit_sha(owner_repo, branch, until_date=None):
    url = f"https://api.github.com/repos/{owner_repo}/commits"
    params = {"sha": branch, "per_page": 1}
    if until_date:
        params["until"] = until_date

    r = requests.get(url, headers=HEADERS, params=params, timeout=10)

    if r.status_code == 200:
        commits = r.json()
        return commits[0]["sha"] if commits else None

    print(f"[ERROR] commit fetch failed for {owner_repo}: {r.status_code}")
    return None


def main():
    with open(candidates_file, "r", encoding="utf-8") as f:
        repos = json.load(f)

    # remove if you do not want sampling during debugging
    sampled_repos = repos
    accepted_repos = {}

    for name, info in sampled_repos.items():
        gh_link = info.get("github_link")
        if not gh_link:
            print(f"[SKIP] {name}: no github_link")
            continue

        gh_link = normalize_github_url(gh_link)
        if gh_link is None:
            print(f"[SKIP] {name}: invalid GitHub URL")
            continue

        print(f"\n[PROCESSING] {name} -> {gh_link}")

        repo_metadata = get_repo_metadata(gh_link)
        if repo_metadata is None:
            continue

        owner_repo = repo_metadata["full_name"]
        default_branch = repo_metadata.get("default_branch")

        commit_sha = get_latest_commit_sha(owner_repo, default_branch, until_date="2026-01-01T00:00:00Z")
        if not commit_sha:
            continue

        allowed, spdx = is_allowed_license(repo_metadata)
        if not allowed:
            print(f"[LICENSE] Skipping {gh_link}: {spdx}")
            continue

        accepted_repo_row = {
            "repo_id": repo_metadata["full_name"],
            "repo_name": name,
            "repo_url": gh_link,
            "license": spdx,
            "default_branch": repo_metadata.get("default_branch"),
            "archived": repo_metadata.get("archived", False),
            "fork": repo_metadata.get("fork", False),
            "description": repo_metadata.get("description"),
            "commit_sha": commit_sha,
            "language": repo_metadata.get("language")
        }

        append_jsonl(accepted_repos_file, accepted_repo_row)
        accepted_repos[name] = {accepted_repo_row["repo_url"]}

        contents = get_repo_contents(accepted_repo_row)
        if contents is None:
            continue
        else:
            # Save every file identifier (path + sha) in the root dir to a separate file for later processing.
            for item in contents:
                if item["type"] == "file":
                    file_row = {
                        "repo_id": repo_metadata["full_name"],
                        "file_path": item["path"],
                        "file_sha": item["sha"],
                    }
                    append_jsonl("accepted_repo_files.jsonl", file_row)

        print(f"[OK] {repo_metadata['full_name']} accepted, root entries: {len(contents)}")

    # Then, extract files from the root dir of every (accepted) repo, and save them to a separate file for later processing.
    
if __name__ == "__main__":
    # clear accepted_repo_files.jsonl
    with open("accepted_repo_files.jsonl", "w") as f:
        pass
    with open("accepted_repos.jsonl", "w") as f:
        pass
    with open("file_entities.jsonl", "w") as f:
        pass
    main()