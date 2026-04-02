# Evaluate repository quality from accepted repo/file snapshots
# Includes:
# - file/documentation/comment metrics
# - commit-history metrics
# - combined repo scoring

from __future__ import annotations

import io
import json
import os
import pathlib
import re
import zipfile
from typing import Any

import shutil
import subprocess 

import requests
from dotenv import load_dotenv

import file_metrics


load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN is not set")

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
}

accepted_repos_file = "accepted_repos.jsonl"
accepted_repo_files_file = "accepted_repo_files.jsonl"
file_entities_file = "file_entities.jsonl"
repo_scores_file = "repo_scores.json"
commit_entities_file = "commit_entities.jsonl"


CORE_EXTENSIONS = {".py", ".java"}

EXCLUDED_DIR_NAMES = {
    "test", "tests", "testing",
    "example", "examples",
    "docs", "doc",
    "generated",
    "build",
    "target",
    "venv",
    ".venv",
    ".git",
}

TRIVIAL_COMMIT_MESSAGES = {
    "fix", "update", "changes", "cleanup", "wip",
    "minor fix", "misc", "typo", "format", "merge", "bump"
}

COMMIT_KEYWORDS = {
    "bug", "fix", "refactor", "docs", "api", "parser",
    "serialization", "performance", "cache", "validation",
    "error", "exception", "compatibility", "typing",
    "comment", "docstring", "javadoc", "test"
}

def save_commit_entity(row, path=commit_entities_file):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_jsonl(path: str | pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_repo_map(accepted_repos_path: str | pathlib.Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(accepted_repos_path)
    return {row["repo_id"]: row for row in rows}

def build_repo_files_map(
    accepted_repo_files_path: str | pathlib.Path,
) -> dict[str, list[dict[str, Any]]]:
    rows = load_jsonl(accepted_repo_files_path)
    repo_files: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        repo_id = row["repo_id"]
        repo_files.setdefault(repo_id, []).append(row)

    return repo_files

def build_commit_row(
    repo_id: str,
    commit_sha: str,
    parent_sha: str | None,
    file_path: str,
    message: str,
    patch_text: str | None,
    change_status: str | None,
    additions: int | None,
    deletions: int | None,
    language: str | None = None,
    alignment_confidence: str = "medium",
) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "commit_sha": commit_sha,
        "parent_sha": parent_sha,
        "file_path": file_path,
        "language": language,
        "source_type": "commit_message",
        "entity_type": "changed_file",
        "entity_name": file_path,
        "doc_type": "commit_message",
        "doc_text": message,
        "doc_token_count": len(re.findall(r"\b\w+\b", message or "")),
        "patch_text": patch_text,
        "patch_token_count": len(re.findall(r"\b\w+\b", patch_text or "")),
        "change_status": change_status,
        "additions": additions,
        "deletions": deletions,
        "alignment_confidence": alignment_confidence,
    }

def is_core_file(file_path: str) -> bool:
    path_lower = file_path.lower()

    if not any(path_lower.endswith(ext) for ext in CORE_EXTENSIONS):
        return False

    path_parts = set(path_lower.split("/"))
    if any(dir_name in EXCLUDED_DIR_NAMES for dir_name in path_parts):
        return False

    return True


def core_file_density(repo_id: str, repo_files_map: dict[str, list[dict[str, Any]]]) -> float:
    repo_files = repo_files_map.get(repo_id, [])
    core_files = [row for row in repo_files if is_core_file(row["file_path"])]
    total_source_files = len(repo_files)
    return len(core_files) / total_source_files if total_source_files > 0 else 0.0


# -------------------------
# Frozen file-content loading
# -------------------------

def build_source_text_by_path(
    repo_row: dict[str, Any],
    repo_files_map: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    repo_id = repo_row["repo_id"]
    commit_sha = repo_row.get("commit_sha")

    if not commit_sha:
        raise ValueError(f"Missing commit_sha for repo {repo_id}")

    zip_url = f"https://api.github.com/repos/{repo_id}/zipball/{commit_sha}"
    print(f"[INFO] Downloading zip for {repo_id}...")

    try:
        response = requests.get(zip_url, headers=HEADERS, timeout=60)
        if response.status_code != 200:
            print(f"[ERROR] Failed to download zip for {repo_id}: {response.status_code}")
            return {}

        source_text_by_path: dict[str, str] = {}
        needed_files = {row["file_path"] for row in repo_files_map.get(repo_id, [])}

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            namelist = z.namelist()
            if not namelist:
                return {}

            prefix = namelist[0].split("/")[0] + "/"

            for zip_path in namelist:
                if zip_path.startswith(prefix) and not zip_path.endswith("/"):
                    repo_path = zip_path[len(prefix):]
                    if repo_path in needed_files:
                        try:
                            with z.open(zip_path) as f:
                                b_content = f.read()
                                source_text_by_path[repo_path] = b_content.decode(
                                    "utf-8", errors="replace"
                                )
                        except Exception as e:
                            print(f"[WARN] Failed to read {repo_path} from zip: {e}")

        return source_text_by_path

    except Exception as e:
        print(f"[ERROR] Failed to process zip for {repo_id}: {e}")
        return {}


# -------------------------
# File metrics aggregation
# -------------------------

def aggregate_file_metrics(per_file_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    total_functions = sum(m["total_functions"] for m in per_file_metrics)
    documented_functions = sum(m["documented_functions"] for m in per_file_metrics)

    total_classes = sum(m["total_classes"] for m in per_file_metrics)
    documented_classes = sum(m["documented_classes"] for m in per_file_metrics)

    total_public_functions = sum(m["total_public_functions"] for m in per_file_metrics)
    documented_public_functions = sum(m["documented_public_functions"] for m in per_file_metrics)

    total_public_classes = sum(m["total_public_classes"] for m in per_file_metrics)
    documented_public_classes = sum(m["documented_public_classes"] for m in per_file_metrics)

    total_code_lines = sum(m["total_code_lines"] for m in per_file_metrics)
    total_comment_lines = sum(m["total_comment_lines"] for m in per_file_metrics)
    meaningful_comment_lines = sum(m["meaningful_comment_lines"] for m in per_file_metrics)

    doc_lengths: list[int] = []
    for m in per_file_metrics:
        doc_lengths.extend(m["doc_lengths"])

    files_with_meaningful_comments = sum(
        1 for m in per_file_metrics if m["has_meaningful_comment"]
    )
    total_core_files = len(per_file_metrics)

    return {
        "total_core_files": total_core_files,
        "total_functions": total_functions,
        "documented_functions": documented_functions,
        "function_doc_coverage": (
            documented_functions / total_functions if total_functions else 0.0
        ),
        "total_classes": total_classes,
        "documented_classes": documented_classes,
        "class_doc_coverage": (
            documented_classes / total_classes if total_classes else 0.0
        ),
        "total_public_functions": total_public_functions,
        "documented_public_functions": documented_public_functions,
        "public_function_doc_coverage": (
            documented_public_functions / total_public_functions
            if total_public_functions else 0.0
        ),
        "total_public_classes": total_public_classes,
        "documented_public_classes": documented_public_classes,
        "public_class_doc_coverage": (
            documented_public_classes / total_public_classes
            if total_public_classes else 0.0
        ),
        "avg_doc_length_tokens": (
            sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
        ),
        "total_code_lines": total_code_lines,
        "total_comment_lines": total_comment_lines,
        "meaningful_comment_lines": meaningful_comment_lines,
        "comment_density": (
            total_comment_lines / total_code_lines if total_code_lines else 0.0
        ),
        "meaningful_comment_density": (
            meaningful_comment_lines / total_code_lines if total_code_lines else 0.0
        ),
        "files_with_meaningful_comments": files_with_meaningful_comments,
        "meaningful_comment_file_ratio": (
            files_with_meaningful_comments / total_core_files if total_core_files else 0.0
        ),
    }


# -------------------------
# Score helpers
# -------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def normalize_cap(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return clamp01(value / cap)


# -------------------------
# Documentation evaluation
# -------------------------

def evaluate_documentation(
    repo_row: dict[str, Any],
    repo_files_map: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    repo_id = repo_row["repo_id"]

    source_text_by_path = build_source_text_by_path(repo_row, repo_files_map)

    repo_files = repo_files_map.get(repo_id, [])
    core_files = [row for row in repo_files if is_core_file(row["file_path"])]

    per_file_metrics: list[dict[str, Any]] = []
    missing_files: list[str] = []

    for row in core_files:
        path = row["file_path"]
        source_text = source_text_by_path.get(path)

        if source_text is None:
            missing_files.append(path)
            continue

        if not source_text.strip():
            print(f"[WARN] Skipping empty file: {path}")
            continue

        file_repo_metadata = {
            "repo_id": repo_id,
            "file_path": path,
            "commit_sha": repo_row["commit_sha"],
        }

        metrics = file_metrics.analyze_source_file(path, source_text, file_repo_metadata)
        per_file_metrics.append(metrics)

    aggregated = aggregate_file_metrics(per_file_metrics)

    return {
        "repo_id": repo_id,
        "repo_url": repo_row.get("repo_url"),
        "language": repo_row.get("language"),
        "n_listed_core_files": len(core_files),
        "n_analyzed_core_files": len(per_file_metrics),
        "missing_files": missing_files,
        "metrics": aggregated,
    }


# -------------------------
# Commit evaluation
# -------------------------

def fetch_repo_commits(
    repo_id: str,
    branch: str,
    max_pages: int = 3,
    per_page: int = 100,
) -> list[dict[str, Any]]:
    commits: list[dict[str, Any]] = []
    page = 1

    while page <= max_pages:
        url = f"https://api.github.com/repos/{repo_id}/commits"
        params = {
            "sha": branch,
            "per_page": per_page,
            "page": page,
        }
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)

        if response.status_code != 200:
            print(f"[WARN] Failed to fetch commits for {repo_id}: {response.status_code}")
            break

        batch = response.json()
        if not batch:
            break

        commits.extend(batch)
        page += 1

    return commits


def fetch_commit_detail(repo_id: str, commit_sha: str) -> dict[str, Any] | None:
    url = f"https://api.github.com/repos/{repo_id}/commits/{commit_sha}"
    response = requests.get(url, headers=HEADERS, timeout=30)

    if response.status_code != 200:
        print(f"[WARN] Failed to fetch commit detail for {repo_id}@{commit_sha}: {response.status_code}")
        return None

    return response.json()


def analyze_commit_message(message: str) -> dict[str, Any]:
    text = (message or "").strip().lower()
    tokens = re.findall(r"\b\w+\b", text)

    is_merge = text.startswith("merge ")
    token_len = len(tokens)
    non_trivial = (
        token_len >= 4
        and text not in TRIVIAL_COMMIT_MESSAGES
        and not is_merge
    )
    keyword_hits = sum(1 for kw in COMMIT_KEYWORDS if kw in text)

    return {
        "token_len": token_len,
        "non_trivial": non_trivial,
        "keyword_hits": keyword_hits,
        "is_merge": is_merge,
    }


def commit_touches_core_files(commit_detail: dict[str, Any]) -> tuple[bool, set[str]]:
    touched_core_files: set[str] = set()

    for f in commit_detail.get("files", []):
        filename = f.get("filename", "")
        if is_core_file(filename):
            touched_core_files.add(filename)

    return (len(touched_core_files) > 0, touched_core_files)


def evaluate_commits(
    repo_row: dict[str, Any],
    repo_files_map: dict[str, list[dict[str, Any]]],
    max_commits_to_inspect: int = 40,
) -> dict[str, Any]:
    repo_id = repo_row["repo_id"]
    branch = repo_row.get("default_branch")

    if not branch:
        return {
            "n_commits_seen": 0,
            "n_commits_touching_core_files": 0,
            "avg_commit_message_length_tokens": 0.0,
            "non_trivial_commit_fraction": 0.0,
            "keyword_bonus_avg": 0.0,
            "core_file_commit_coverage": 0.0,
        }

    core_files = {
        row["file_path"]
        for row in repo_files_map.get(repo_id, [])
        if is_core_file(row["file_path"])
    }

    commits = fetch_repo_commits(repo_id, branch)
    commits = commits[:max_commits_to_inspect]

    commit_lengths: list[int] = []
    non_trivial_count = 0
    keyword_hits_total = 0
    touched_core_files_union: set[str] = set()
    n_commits_touching_core_files = 0

    for commit in commits:
        sha = commit.get("sha")
        if not sha:
            continue

        detail = fetch_commit_detail(repo_id, sha)
        if not detail:
            continue

        message = detail.get("commit", {}).get("message", "")
        parent_sha = None
        parents = detail.get("parents", [])
        if parents:
            parent_sha = parents[0].get("sha")

        touched_any_core = False
        touched_files_this_commit = set()

        for f in detail.get("files", []):
            filename = f.get("filename", "")
            if not is_core_file(filename):
                continue

            touched_any_core = True
            touched_files_this_commit.add(filename)

            patch_text = f.get("patch")  # may be missing for large/binary files
            change_status = f.get("status")
            additions = f.get("additions")
            deletions = f.get("deletions")

            language = None
            if filename.endswith(".py"):
                language = "python"
            elif filename.endswith(".java"):
                language = "java"

            row = build_commit_row(
                repo_id=repo_id,
                commit_sha=sha,
                parent_sha=parent_sha,
                file_path=filename,
                message=message,
                patch_text=patch_text,
                change_status=change_status,
                additions=additions,
                deletions=deletions,
                language=language,
                alignment_confidence="medium" if patch_text else "low",
            )
            save_commit_entity(row)

        if not touched_any_core:
            continue

        n_commits_touching_core_files += 1
        touched_core_files_union.update(touched_files_this_commit)

        msg_stats = analyze_commit_message(message)
        commit_lengths.append(msg_stats["token_len"])
        if msg_stats["non_trivial"]:
            non_trivial_count += 1
        keyword_hits_total += msg_stats["keyword_hits"]

    avg_commit_len = (
        sum(commit_lengths) / len(commit_lengths) if commit_lengths else 0.0
    )
    non_trivial_fraction = (
        non_trivial_count / n_commits_touching_core_files
        if n_commits_touching_core_files else 0.0
    )
    keyword_bonus_avg = (
        keyword_hits_total / n_commits_touching_core_files
        if n_commits_touching_core_files else 0.0
    )
    core_file_commit_coverage = (
        len(touched_core_files_union & core_files) / len(core_files)
        if core_files else 0.0
    )

    return {
        "n_commits_seen": len(commits),
        "n_commits_touching_core_files": n_commits_touching_core_files,
        "avg_commit_message_length_tokens": avg_commit_len,
        "non_trivial_commit_fraction": non_trivial_fraction,
        "keyword_bonus_avg": keyword_bonus_avg,
        "core_file_commit_coverage": core_file_commit_coverage,
    }


# -------------------------
# Combined scoring
# -------------------------

def score_commit_metrics(commit_metrics: dict[str, Any]) -> dict[str, Any]:
    norm_commit_count = normalize_cap(commit_metrics["n_commits_touching_core_files"], 200.0)
    norm_commit_len = normalize_cap(commit_metrics["avg_commit_message_length_tokens"], 12.0)
    norm_keyword_bonus = normalize_cap(commit_metrics["keyword_bonus_avg"], 2.0)

    commit_score = (
        0.35 * commit_metrics["non_trivial_commit_fraction"] +
        0.25 * norm_commit_len +
        0.20 * norm_commit_count +
        0.10 * norm_keyword_bonus +
        0.10 * commit_metrics["core_file_commit_coverage"]
    )

    return {
        "commit_score": commit_score,
    }


def score_repo(
    doc_metrics: dict[str, Any],
    commit_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    norm_avg_doc_length = normalize_cap(doc_metrics["avg_doc_length_tokens"], 40.0)
    norm_comment_density = normalize_cap(doc_metrics["comment_density"], 0.20)
    norm_meaningful_comment_density = normalize_cap(
        doc_metrics["meaningful_comment_density"], 0.10
    )

    api_doc_score = (
        0.35 * doc_metrics["public_function_doc_coverage"] +
        0.15 * doc_metrics["function_doc_coverage"] +
        0.20 * doc_metrics["public_class_doc_coverage"] +
        0.10 * doc_metrics["class_doc_coverage"] +
        0.20 * norm_avg_doc_length
    )

    comment_score = (
        0.45 * doc_metrics["meaningful_comment_file_ratio"] +
        0.35 * norm_meaningful_comment_density +
        0.20 * norm_comment_density
    )

    doc_comment_score = 0.70 * api_doc_score + 0.30 * comment_score

    if commit_metrics is None:
        commit_score = 0.0
        final_score = doc_comment_score
    else:
        commit_score = score_commit_metrics(commit_metrics)["commit_score"]
        final_score = 0.65 * doc_comment_score + 0.35 * commit_score

    if final_score >= 0.75:
        bucket = "high"
    elif final_score >= 0.55:
        bucket = "medium"
    elif final_score >= 0.35:
        bucket = "borderline"
    else:
        bucket = "reject"

    return {
        "api_doc_score": api_doc_score,
        "comment_score": comment_score,
        "commit_score": commit_score,
        "final_score": final_score,
        "quality_bucket": bucket,
    }


# -------------------------
# Main
# -------------------------

def main():
    repo_map = build_repo_map(accepted_repos_file)
    repo_files_map = build_repo_files_map(accepted_repo_files_file)
    all_repo_scores = []

    for repo_id, repo_row in repo_map.items():
        repo_url = repo_row.get("repo_url")
        if not repo_url:
            print(f"[SKIP] {repo_id}: missing repo_url")
            continue

        if not repo_row.get("commit_sha"):
            print(f"[SKIP] {repo_id}: missing commit_sha")
            continue

        try:
            core_density = core_file_density(repo_id, repo_files_map)
            doc_result = evaluate_documentation(repo_row, repo_files_map)
            commit_result = evaluate_commits(repo_row, repo_files_map)
            scores = score_repo(doc_result["metrics"], commit_result)
        except Exception as e:
            print(f"[ERROR] {repo_id}: {e}")
            print("-" * 40)
            continue

        metrics = doc_result["metrics"]

        print(f"Repo: {repo_url}")
        print(f"Core file density: {core_density:.2f}")
        print(
            f"Documentation: "
            f"function={metrics['function_doc_coverage']:.2f}, "
            f"public_function={metrics['public_function_doc_coverage']:.2f}, "
            f"class={metrics['class_doc_coverage']:.2f}, "
            f"public_class={metrics['public_class_doc_coverage']:.2f}"
        )
        print(f"Average doc length: {metrics['avg_doc_length_tokens']:.2f}")
        print(
            f"Comments: density={metrics['comment_density']:.3f}, "
            f"meaningful_density={metrics['meaningful_comment_density']:.3f}, "
            f"meaningful_file_ratio={metrics['meaningful_comment_file_ratio']:.2f}"
        )
        print(
            f"Commits: "
            f"seen={commit_result['n_commits_seen']}, "
            f"touching_core={commit_result['n_commits_touching_core_files']}, "
            f"avg_len={commit_result['avg_commit_message_length_tokens']:.2f}, "
            f"non_trivial={commit_result['non_trivial_commit_fraction']:.2f}, "
            f"keyword_bonus={commit_result['keyword_bonus_avg']:.2f}, "
            f"core_file_coverage={commit_result['core_file_commit_coverage']:.2f}"
        )
        print(
            f"Score: final={scores['final_score']:.2f}, "
            f"api={scores['api_doc_score']:.2f}, "
            f"comment={scores['comment_score']:.2f}, "
            f"commit={scores['commit_score']:.2f}, "
            f"bucket={scores['quality_bucket']}"
        )
        if doc_result["missing_files"]:
            print(f"Missing files fetched: {len(doc_result['missing_files'])}")

        result_row = {
            **doc_result,
            "core_density": core_density,
            "commit_metrics": commit_result,
            "scores": scores,
        }
        all_repo_scores.append(result_row)

        print("-" * 40)

    print(f"[INFO] Saving all scores to {repo_scores_file}...")
    with open(repo_scores_file, "w", encoding="utf-8") as f:
        json.dump(all_repo_scores, f, indent=2)
    print("[SUCCESS] Done!")


if __name__ == "__main__":
    with open(file_entities_file, "w", encoding="utf-8") as f:
        f.write("")
    with open(commit_entities_file, "w", encoding="utf-8") as f:
        f.write("")
    with open(repo_scores_file, "w", encoding="utf-8") as f:
        f.write("")
    main()