# Get a list of repositories from PyPI, Maven central and Awesome lists, 
# then save the corresponding GH links to a file for later use

import requests
import json
import random
import re
from pathlib import Path

def normalize_gh_link(link):
    ### 
    # Normalize GitHub link to owner/repo format
    ###
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




def get_pypi_repos():
    ###
    # Get top packages from https://github.com/hugovk/top-pypi-packages/tree/main 
    # In format: 
    # pkg_info = {
    #                 "name": pkg_data["info"]["name"],
    #                 "version": pkg_data["info"]["version"],
    #                 "summary": pkg_data["info"]["summary"],
    #                 "license": pkg_data["info"]["license"],
    #                 "project_urls": pkg_data["info"].get("project_urls"),
    #             }
    ###

    url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json"
    response = requests.get(url)
    top_packages = response.json().get("rows", [])[:500]  # Get top 500 packages
    top_package_names = [pkg["project"] for pkg in top_packages]   

    # Find them in PyPI and get their GH links via metadata

    output_dir = Path("/tmp/pypi_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    gh_links = {}

    for pkg in top_package_names:
        pkg_url = f"https://pypi.org/pypi/{pkg}/json"
        
        try:
            response = requests.get(pkg_url, timeout=10)

            if response.status_code != 200:
                print(f"[ERROR] {pkg}: status {response.status_code}")
                continue

            pkg_data = response.json()

            # If homepage is a GH link, use it; otherwise look for GH link in project_urls (under "Source" or "Code" or "Home"/"Homepage" or "Repository")
            gh_link = None
            homepage = pkg_data["info"].get("home_page", "")
            if homepage and "github.com" in homepage.lower():
                gh_link = homepage
            else:
                project_urls = pkg_data["info"].get("project_urls", {})
                for key in ["Source", "Code", "Home", "Homepage", "Repository"]:
                    url = project_urls.get(key, "")
                    if url and "github.com" in url.lower():
                        gh_link = url
                        break
            
            # Check if there's a changelog
            changelog_url = None
            for key in ["Changelog", "Change Log", "Changes"]:
                changelog_url = project_urls.get(key, "")
                if changelog_url:
                    print(f"{pkg} has a changelog: {changelog_url}")
                    break
            
            if gh_link:
                gh_links[pkg] = {"github_link": gh_link, "description": pkg_data["info"]["summary"], "changelog_url": changelog_url, "license": pkg_data["info"]["license"], "source": "pypi"}
                print("pkg name:", pkg, "GitHub link:", gh_link)

        except Exception as e:
            print(f"[EXCEPTION] {pkg}: {e}")

    return gh_links

def extract_entries(markdown_text):
    ###
    # Extract entries from markdown content in the format:
    # - [Repo Name](GitHub Link) - Description
    ###
    entries = []

    entry_pattern = re.compile(
    r'^- \[(?P<name>[^\]]+)\]\((?P<link>[^)]+)\)\s*-\s*(?P<description>.+)$'
    )

    for line in markdown_text.splitlines():
        line = line.strip()

        match = entry_pattern.match(line)
        if not match:
            continue

        link = match.group("link").strip()

        # keep only GitHub repos
        if "github.com" not in link.lower():
            continue

        entries.append({
            "name": match.group("name").strip(),
            "github_link": link,
            "description": match.group("description").strip(),
        })

    return entries


def get_awesome_repos():
    ###
    # Get awesome lists from Awesome Python and Awesome Java
    # {'name': , 
    #   'github_link': , 
    #   'description': 
    # }
    ###
    awesome_python_url = "https://raw.githubusercontent.com/vinta/awesome-python/master/README.md"
    awesome_java_url = "https://raw.githubusercontent.com/akullpp/awesome-java/master/README.md"

    gh_links = {}

    # Extract entries from markdown content
    for url in [awesome_python_url, awesome_java_url]:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"[ERROR] Failed to fetch {url}: status {response.status_code}")
            continue

        lang = "Python" if "python" in url.lower() else "Java"

        entries = extract_entries(response.text)
        for entry in entries:
            gh_links[entry["name"]] = {"github_link": entry["github_link"], "description": entry["description"], "changelog_url": None, "license": None, "source": f"awesome-{lang.lower()}"}

    return gh_links

def get_all_repos():
    ###
    # To ensure variety, we will sample a balanced mix of repositories from:
    # 1) Top PyPI packages (100 repos)
    # 2) Awesome Python (50 repos)
    # 3) Awesome Java (150 repos)
    ###
    print("[INFO] Fetching PyPI repositories (top 500)...")
    pypi_repos = get_pypi_repos()
    
    print("[INFO] Fetching Awesome Python/Java repositories...")
    awesome_repos = get_awesome_repos()

    # Split awesome repos by source for balanced sampling
    awesome_py = {k: v for k, v in awesome_repos.items() if v["source"] == "awesome-python"}
    awesome_java = {k: v for k, v in awesome_repos.items() if v["source"] == "awesome-java"}

    # Targets for variety and balance
    TARGET_PYPI = 100
    TARGET_AWESOME_PY = 50
    TARGET_AWESOME_JAVA = 150

    # Perform random sampling
    random.seed(42)  # For reproducibility
    
    sampled_pypi = dict(random.sample(list(pypi_repos.items()), min(len(pypi_repos), TARGET_PYPI)))
    sampled_awesome_py = dict(random.sample(list(awesome_py.items()), min(len(awesome_py), TARGET_AWESOME_PY)))
    sampled_awesome_java = dict(random.sample(list(awesome_java.items()), min(len(awesome_java), TARGET_AWESOME_JAVA)))

    print(f"[INFO] Sampled {len(sampled_pypi)} from PyPI")
    print(f"[INFO] Sampled {len(sampled_awesome_py)} from Awesome-Python")
    print(f"[INFO] Sampled {len(sampled_awesome_java)} from Awesome-Java")

    all_repos = {**sampled_pypi, **sampled_awesome_py, **sampled_awesome_java}

    # deduplicate by normalized repo identity (owner/repo)
    seen_repos = set()
    deduped_repos = {}
    for name, info in all_repos.items():
        repo_id = normalize_gh_link(info["github_link"])
        if repo_id and repo_id not in seen_repos:
            deduped_repos[name] = info
            seen_repos.add(repo_id)
        
    print(f"[SUCCESS] Final dataset variety: {len(deduped_repos)} unique repositories")

    return deduped_repos


if __name__ == "__main__":
    repos = get_all_repos()

    with open("candidate_repos.json", "w") as f:
        json.dump(repos, f, indent=2)
    print("[OK] Saved to candidate_repos.json")