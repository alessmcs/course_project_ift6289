# load files from repo & compute per-file metrics

import io
import re
import os
import json
import tokenize
import ast

from tree_sitter_language_pack import get_parser
from typing import Any
from dotenv import load_dotenv

file_entities_file = "file_entities.jsonl"

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN is not set")

JAVA_PARSER = get_parser("java")

TRIVIAL_COMMENT_PATTERNS = [
    r"^\s*$",
    r"^\s*(TODO|todo|fixme|xxx|hack)\b[:\s-]*.*$",
    r"^\s*(noinspection|nosonar|pylint|type:\s*ignore)\b.*$",
    r"^\s*[-=*_/\\#]{3,}\s*$",
]


def count_tokens(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def is_meaningful_comment(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    if count_tokens(t) < 3:
        return False
    if re.fullmatch(r"[\W_]+", t):
        return False
    for pat in TRIVIAL_COMMENT_PATTERNS:
        if re.match(pat, t):
            return False
    return True


def build_row(
    repo_id,
    file_path,
    commit_sha,
    language,
    entity_type,
    entity_name,
    source_type,
    doc_text,
    doc_start_line,
    doc_end_line,
    entity_start_line,
    entity_end_line,
    doc_start_col=None,
    doc_end_col=None,
    entity_start_col=None,
    entity_end_col=None,
    qualified_name=None,
    alignment_confidence="high",
    extra=None,
):
    row = {
        "repo_id": repo_id,
        "file_path": file_path,
        "commit_sha": commit_sha,
        "language": language,
        "entity_type": entity_type,
        "entity_name": entity_name,
        "qualified_name": qualified_name,
        "source_type": source_type,
        "doc_text": doc_text,
        "doc_token_count": count_tokens(doc_text) if doc_text else 0,
        "doc_start_line": doc_start_line,
        "doc_end_line": doc_end_line,
        "doc_start_col": doc_start_col,
        "doc_end_col": doc_end_col,
        "entity_start_line": entity_start_line,
        "entity_end_line": entity_end_line,
        "entity_start_col": entity_start_col,
        "entity_end_col": entity_end_col,
        "alignment_confidence": alignment_confidence,
    }
    if extra:
        row.update(extra)
    return row


def save_entity(row, path=file_entities_file):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------
# Python analysis
# --------------------

def extract_python_comments(source_text):
    comments = []
    reader = io.StringIO(source_text).readline
    for tok in tokenize.generate_tokens(reader):
        if tok.type == tokenize.COMMENT:
            comments.append({
                "start_line": tok.start[0],
                "start_col": tok.start[1],
                "end_line": tok.end[0],
                "end_col": tok.end[1],
                "text": tok.string.lstrip("#").strip(),
            })
    return comments


def analyze_python_file(source_text, repo_metadata):
    repo_id = repo_metadata.get("repo_id")
    file_path = repo_metadata.get("file_path")
    commit_sha = repo_metadata.get("commit_sha")
    language = "Python"

    total_functions = 0
    documented_functions = 0
    total_classes = 0
    documented_classes = 0
    total_public_functions = 0
    documented_public_functions = 0
    total_public_classes = 0
    documented_public_classes = 0
    doc_lengths: list[int] = []

    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return {
            "total_functions": 0,
            "documented_functions": 0,
            "total_classes": 0,
            "documented_classes": 0,
            "total_public_functions": 0,
            "documented_public_functions": 0,
            "total_public_classes": 0,
            "documented_public_classes": 0,
            "doc_lengths": [],
            "total_code_lines": 0,
            "total_comment_lines": 0,
            "meaningful_comment_lines": 0,
            "has_meaningful_comment": False,
        }

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            total_functions += 1
            is_public = not node.name.startswith("_")
            if is_public:
                total_public_functions += 1

            node_start_line = getattr(node, "lineno", None)
            node_end_line = getattr(node, "end_lineno", node_start_line)
            node_start_col = getattr(node, "col_offset", None)
            node_end_col = getattr(node, "end_col_offset", None)

            doc = ast.get_docstring(node)
            if doc and node.body:
                documented_functions += 1
                doc_lengths.append(count_tokens(doc))
                if is_public:
                    documented_public_functions += 1

                first_stmt = node.body[0]
                doc_start_line = getattr(first_stmt, "lineno", None)
                doc_end_line = getattr(first_stmt, "end_lineno", doc_start_line)
                doc_start_col = getattr(first_stmt, "col_offset", None)
                doc_end_col = getattr(first_stmt, "end_col_offset", None)

                row = build_row(
                    repo_id=repo_id,
                    file_path=file_path,
                    commit_sha=commit_sha,
                    language=language,
                    entity_type="function",
                    entity_name=node.name,
                    source_type="docstring",
                    doc_text=doc,
                    doc_start_line=doc_start_line,
                    doc_end_line=doc_end_line,
                    entity_start_line=node_start_line,
                    entity_end_line=node_end_line,
                    doc_start_col=doc_start_col,
                    doc_end_col=doc_end_col,
                    entity_start_col=node_start_col,
                    entity_end_col=node_end_col,
                )
                save_entity(row)

        elif isinstance(node, ast.ClassDef):
            total_classes += 1
            is_public = not node.name.startswith("_")
            if is_public:
                total_public_classes += 1

            node_start_line = getattr(node, "lineno", None)
            node_end_line = getattr(node, "end_lineno", node_start_line)
            node_start_col = getattr(node, "col_offset", None)
            node_end_col = getattr(node, "end_col_offset", None)

            doc = ast.get_docstring(node)
            if doc and node.body:
                documented_classes += 1
                doc_lengths.append(count_tokens(doc))
                if is_public:
                    documented_public_classes += 1

                first_stmt = node.body[0]
                doc_start_line = getattr(first_stmt, "lineno", None)
                doc_end_line = getattr(first_stmt, "end_lineno", doc_start_line)
                doc_start_col = getattr(first_stmt, "col_offset", None)
                doc_end_col = getattr(first_stmt, "end_col_offset", None)

                row = build_row(
                    repo_id=repo_id,
                    file_path=file_path,
                    commit_sha=commit_sha,
                    language=language,
                    entity_type="class",
                    entity_name=node.name,
                    source_type="docstring",
                    doc_text=doc,
                    doc_start_line=doc_start_line,
                    doc_end_line=doc_end_line,
                    entity_start_line=node_start_line,
                    entity_end_line=node_end_line,
                    doc_start_col=doc_start_col,
                    doc_end_col=doc_end_col,
                    entity_start_col=node_start_col,
                    entity_end_col=node_end_col,
                )
                save_entity(row)

    lines = source_text.splitlines()
    total_code_lines = sum(1 for line in lines if line.strip())

    comments = extract_python_comments(source_text)
    for comment in comments:
        row = build_row(
            repo_id=repo_id,
            file_path=file_path,
            commit_sha=commit_sha,
            language=language,
            entity_type="comment",
            entity_name=None,
            source_type="inline_comment",
            doc_text=comment["text"],
            doc_start_line=comment["start_line"],
            doc_end_line=comment["end_line"],
            entity_start_line=comment["start_line"],
            entity_end_line=comment["end_line"],
            doc_start_col=comment["start_col"],
            doc_end_col=comment["end_col"],
            entity_start_col=comment["start_col"],
            entity_end_col=comment["end_col"],
            alignment_confidence="medium" if comment["start_col"] == 0 else "high",
        )
        save_entity(row)

    total_comment_lines = len(comments)
    meaningful_comments = [c for c in comments if is_meaningful_comment(c["text"])]
    meaningful_comment_lines = len(meaningful_comments)

    return {
        "total_functions": total_functions,
        "documented_functions": documented_functions,
        "total_classes": total_classes,
        "documented_classes": documented_classes,
        "total_public_functions": total_public_functions,
        "documented_public_functions": documented_public_functions,
        "total_public_classes": total_public_classes,
        "documented_public_classes": documented_public_classes,
        "doc_lengths": doc_lengths,
        "total_code_lines": total_code_lines,
        "total_comment_lines": total_comment_lines,
        "meaningful_comment_lines": meaningful_comment_lines,
        "has_meaningful_comment": meaningful_comment_lines > 0,
    }


# --------------------
# Java helpers
# --------------------

def walk_tree(node):
    yield node
    for child in node.children:
        yield from walk_tree(child)


def node_text(node, source_bytes):
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def child_by_field_name(node, field_name):
    return node.child_by_field_name(field_name)


def clean_javadoc_text(raw):
    text = raw.strip()

    if text.startswith("/**"):
        text = text[3:]
    elif text.startswith("/*"):
        text = text[2:]

    if text.endswith("*/"):
        text = text[:-2]

    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\*\s?", "", line)
        lines.append(line.strip())

    return "\n".join(lines).strip()


def clean_java_comment_text(raw):
    text = raw.strip()

    if text.startswith("//"):
        text = text[2:]
    elif text.startswith("/*"):
        text = text[2:]

    if text.endswith("*/"):
        text = text[:-2]

    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\*?\s?", "", line)
        lines.append(line.strip())

    return "\n".join(lines).strip()


def get_modifiers_text(node, source_bytes):
    modifiers = child_by_field_name(node, "modifiers")
    if modifiers is None:
        return ""
    return node_text(modifiers, source_bytes)


def is_public_declaration(node, source_bytes):
    return "public" in get_modifiers_text(node, source_bytes).split()


def previous_sibling(node):
    return node.prev_sibling


def find_attached_javadoc(node, source_bytes):
    sib = previous_sibling(node)

    while sib is not None:
        if sib.type in {"comment", "line_comment", "block_comment"}:
            raw = node_text(sib, source_bytes)
            if raw.lstrip().startswith("/**"):
                comment_end_line = sib.end_point[0] + 1
                decl_start_line = node.start_point[0] + 1

                if decl_start_line - comment_end_line <= 1:
                    return {
                        "text": clean_javadoc_text(raw),
                        "start_line": sib.start_point[0] + 1,
                        "end_line": sib.end_point[0] + 1,
                        "start_col": sib.start_point[1],
                        "end_col": sib.end_point[1],
                    }
            break

        if sib.type in {"modifiers", "marker_annotation", "annotation"}:
            sib = previous_sibling(sib)
            continue

        break

    return None


def extract_java_comments(source_text):
    source_bytes = source_text.encode("utf-8")
    tree = JAVA_PARSER.parse(source_bytes)
    root = tree.root_node

    comments = []

    for node in walk_tree(root):
        if node.type not in {"comment", "line_comment", "block_comment"}:
            continue

        raw = node_text(node, source_bytes)
        stripped = raw.lstrip()
        is_javadoc = stripped.startswith("/**")
        cleaned = clean_javadoc_text(raw) if is_javadoc else clean_java_comment_text(raw)

        comments.append({
            "text": cleaned,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "start_col": node.start_point[1],
            "end_col": node.end_point[1],
            "is_javadoc": is_javadoc,
            "is_meaningful": is_meaningful_comment(cleaned),
        })

    return comments


# --------------------
# Java analysis
# --------------------

def analyze_java_file(source_text: str, repo_metadata) -> dict[str, Any]:
    repo_id = repo_metadata.get("repo_id")
    file_path = repo_metadata.get("file_path")
    commit_sha = repo_metadata.get("commit_sha")
    language = "Java"

    total_functions = 0
    documented_functions = 0
    total_classes = 0
    documented_classes = 0
    total_public_functions = 0
    documented_public_functions = 0
    total_public_classes = 0
    documented_public_classes = 0
    doc_lengths: list[int] = []

    if not source_text or len(source_text.strip()) == 0:
        return {
            "total_functions": 0,
            "documented_functions": 0,
            "total_classes": 0,
            "documented_classes": 0,
            "total_public_functions": 0,
            "documented_public_functions": 0,
            "total_public_classes": 0,
            "documented_public_classes": 0,
            "doc_lengths": [],
            "total_code_lines": 0,
            "total_comment_lines": 0,
            "meaningful_comment_lines": 0,
            "has_meaningful_comment": False,
        }

    source_bytes = source_text.encode("utf-8")

    try:
        tree = JAVA_PARSER.parse(source_bytes)
    except Exception as e:
        return {
            "total_functions": 0,
            "documented_functions": 0,
            "total_classes": 0,
            "documented_classes": 0,
            "total_public_functions": 0,
            "documented_public_functions": 0,
            "total_public_classes": 0,
            "documented_public_classes": 0,
            "doc_lengths": [],
            "total_code_lines": 0,
            "total_comment_lines": 0,
            "meaningful_comment_lines": 0,
            "has_meaningful_comment": False,
        }

    root = tree.root_node
    nodes_visited = 0

    for node in walk_tree(root):
        nodes_visited += 1
        node_type = node.type
        node_start_line = node.start_point[0] + 1
        node_end_line = node.end_point[0] + 1
        node_start_col = node.start_point[1]
        node_end_col = node.end_point[1]

        if node_type in {"method_declaration", "constructor_declaration", "method_definition"}:
            total_functions += 1

            name_node = child_by_field_name(node, "name")
            entity_name = node_text(name_node, source_bytes) if name_node else None

            is_public = is_public_declaration(node, source_bytes)
            if is_public:
                total_public_functions += 1

            javadoc = find_attached_javadoc(node, source_bytes)
            if javadoc and javadoc["text"]:
                documented_functions += 1
                doc_lengths.append(count_tokens(javadoc["text"]))

                if is_public:
                    documented_public_functions += 1

                entity_type = "constructor" if node_type == "constructor_declaration" else "function"

                row = build_row(
                    repo_id=repo_id,
                    file_path=file_path,
                    commit_sha=commit_sha,
                    language=language,
                    entity_type=entity_type,
                    entity_name=entity_name,
                    source_type="javadoc",
                    doc_text=javadoc["text"],
                    doc_start_line=javadoc["start_line"],
                    doc_end_line=javadoc["end_line"],
                    entity_start_line=node_start_line,
                    entity_end_line=node_end_line,
                    doc_start_col=javadoc["start_col"],
                    doc_end_col=javadoc["end_col"],
                    entity_start_col=node_start_col,
                    entity_end_col=node_end_col,
                )
                save_entity(row)

        elif node_type in {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        } or (node_type == "class" and node.parent and node.parent.type != "class_declaration") \
          or (node_type == "interface" and node.parent and node.parent.type != "interface_declaration"):
            # Avoid counting the keyword itself if the parent is already the declaration
            total_classes += 1

            name_node = child_by_field_name(node, "name")
            entity_name = node_text(name_node, source_bytes) if name_node else None

            is_public = is_public_declaration(node, source_bytes)
            if is_public:
                total_public_classes += 1

            javadoc = find_attached_javadoc(node, source_bytes)
            if javadoc and javadoc["text"]:
                documented_classes += 1
                doc_lengths.append(count_tokens(javadoc["text"]))

                if is_public:
                    documented_public_classes += 1

                if node_type == "interface_declaration":
                    entity_type = "interface"
                elif node_type == "enum_declaration":
                    entity_type = "enum"
                elif node_type == "record_declaration":
                    entity_type = "record"
                else:
                    entity_type = "class"

                row = build_row(
                    repo_id=repo_id,
                    file_path=file_path,
                    commit_sha=commit_sha,
                    language=language,
                    entity_type=entity_type,
                    entity_name=entity_name,
                    source_type="javadoc",
                    doc_text=javadoc["text"],
                    doc_start_line=javadoc["start_line"],
                    doc_end_line=javadoc["end_line"],
                    entity_start_line=node_start_line,
                    entity_end_line=node_end_line,
                    doc_start_col=javadoc["start_col"],
                    doc_end_col=javadoc["end_col"],
                    entity_start_col=node_start_col,
                    entity_end_col=node_end_col,
                )
                save_entity(row)

    lines = source_text.splitlines()
    total_code_lines = sum(1 for line in lines if line.strip())

    comments = extract_java_comments(source_text)
    non_javadoc_comments = [c for c in comments if not c["is_javadoc"]]

    for comment in non_javadoc_comments:
        row = build_row(
            repo_id=repo_id,
            file_path=file_path,
            commit_sha=commit_sha,
            language=language,
            entity_type="comment",
            entity_name=None,
            source_type="inline_comment",
            doc_text=comment["text"],
            doc_start_line=comment["start_line"],
            doc_end_line=comment["end_line"],
            entity_start_line=comment["start_line"],
            entity_end_line=comment["end_line"],
            doc_start_col=comment["start_col"],
            doc_end_col=comment["end_col"],
            entity_start_col=comment["start_col"],
            entity_end_col=comment["end_col"],
            alignment_confidence="medium" if comment["start_col"] == 0 else "high",
        )
        save_entity(row)

    total_comment_lines = sum(
        comment["end_line"] - comment["start_line"] + 1
        for comment in non_javadoc_comments
    )

    meaningful_comments = [c for c in non_javadoc_comments if c["is_meaningful"]]
    meaningful_comment_lines = sum(
        comment["end_line"] - comment["start_line"] + 1
        for comment in meaningful_comments
    )

    if total_functions == 0 and total_classes == 0 and total_code_lines > 20:
        pass

    return {
        "total_functions": total_functions,
        "documented_functions": documented_functions,
        "total_classes": total_classes,
        "documented_classes": documented_classes,
        "total_public_functions": total_public_functions,
        "documented_public_functions": documented_public_functions,
        "total_public_classes": total_public_classes,
        "documented_public_classes": documented_public_classes,
        "doc_lengths": doc_lengths,
        "total_code_lines": total_code_lines,
        "total_comment_lines": total_comment_lines,
        "meaningful_comment_lines": meaningful_comment_lines,
        "has_meaningful_comment": meaningful_comment_lines > 0,
    }

def analyze_source_file(path, source_text, repo_metadata):
    if path.endswith(".py"):
        return analyze_python_file(source_text, repo_metadata)
    elif path.endswith(".java"):
        return analyze_java_file(source_text, repo_metadata)
    else:
        return {
        "total_functions": 0,
        "documented_functions": 0,
        "total_classes": 0,
        "documented_classes": 0,
        "total_public_functions": 0,
        "documented_public_functions": 0,
        "total_public_classes": 0,
        "documented_public_classes": 0,
        "doc_lengths": [],
        "total_code_lines": 0,
        "total_comment_lines": 0,
        "meaningful_comment_lines": 0,
        "has_meaningful_comment": False,
    }