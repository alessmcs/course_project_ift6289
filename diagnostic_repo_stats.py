import json

def normalize_gh_link(link):
    if not link:
        return None
    link = link.strip().rstrip("/")
    if "github.com/" not in link:
        return None
    
    # Extract owner/repo
    parts = link.split("github.com/")[1].split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}".lower().replace(".git", "")
    return None

def diagnostic():
    with open("candidate_repos.json", "r") as f:
        candidates = json.load(f)
    
    seen_repos = set()
    license_skipped = []
    deduped_out = []
    api_errors = []
    accepted = []

    print(f"Total entries in candidate_repos.json: {len(candidates)}")
    
    with open("accepted_repos.jsonl", "r") as f:
        accepted_ids = {json.loads(line)["repo_id"] for line in f}

    for name, data in candidates.items():
        link = data.get("github_link")
        repo_id = normalize_gh_link(link)
        
        if not repo_id:
            api_errors.append(name)
            continue
            
        if repo_id in seen_repos:
            deduped_out.append(repo_id)
            continue
        
        seen_repos.add(repo_id)
        
        if repo_id in accepted_ids:
            accepted.append(repo_id)
        else:
            license_skipped.append(repo_id)

    print(f"Accepted unique repos: {len(accepted)}")
    print(f"Duplicate repo links in candidates: {len(deduped_out)}")
    print(f"Rejected/Skipped unique repos: {len(license_skipped)}")
    print(f"Invalid links: {len(api_errors)}")

if __name__ == "__main__":
    diagnostic()
