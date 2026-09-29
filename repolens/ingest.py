"""Phase 1 - Ingest: download a repository and collect its source files.

Usage:
    python -m repolens.ingest https://github.com/owner/repo
    python -m repolens.ingest D:/some/local/folder
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Downloaded repos are cached here so we only clone each one once.
REPOS_DIR = Path(__file__).resolve().parent.parent / "data" / "repos"

# Folders that never contain code worth reading (dependencies, build output, caches).
IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build",
    ".next", ".idea", ".vscode", "vendor", "target", ".mypy_cache", ".pytest_cache",
    ".tox", "coverage", "site-packages",
}

# Auto-generated files: huge, and they explain nothing about how the code works.
IGNORED_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Cargo.lock", "composer.lock", "Pipfile.lock", "uv.lock",
}

# File extension -> language. Anything not listed here is skipped.
LANGUAGES = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".java": "Java", ".c": "C", ".h": "C", ".cpp": "C++",
    ".hpp": "C++", ".cc": "C++", ".cs": "C#", ".go": "Go", ".rs": "Rust", ".rb": "Ruby",
    ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin", ".sh": "Shell", ".sql": "SQL",
    ".html": "HTML", ".css": "CSS", ".scss": "CSS", ".vue": "Vue", ".svelte": "Svelte",
    ".md": "Markdown", ".rst": "reStructuredText", ".toml": "Config", ".yaml": "Config",
    ".yml": "Config", ".json": "Config", ".ini": "Config", ".cfg": "Config",
}
SPECIAL_FILES = {"Dockerfile": "Docker", "Makefile": "Make"}

# Files bigger than this are usually generated or data dumps, not hand-written code.
MAX_FILE_BYTES = 200_000


@dataclass
class SourceFile:
    path: str      # relative to the repo root, always with forward slashes
    language: str
    text: str
    lines: int


def clone_repo(url: str) -> Path:
    """Shallow-clone a GitHub repo into data/repos/ (reuses an existing copy)."""
    owner, repo = url.rstrip("/").removesuffix(".git").split("/")[-2:]
    dest = REPOS_DIR / f"{owner}__{repo}"
    if dest.exists():
        print(f"Using cached copy: {dest}")
        return dest
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {url} ...")
    # --depth 1 downloads only the latest version, not the whole history.
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    return dest


def resolve_source(target: str) -> Path:
    """Accept either a local folder or a git URL."""
    if Path(target).is_dir():
        return Path(target).resolve()
    return clone_repo(target)


def detect_language(path: Path) -> str | None:
    return SPECIAL_FILES.get(path.name) or LANGUAGES.get(path.suffix.lower())


def collect_files(root: Path) -> tuple[list[SourceFile], Counter]:
    """Walk the repo and return the readable source files, plus why others were skipped."""
    files: list[SourceFile] = []
    skipped: Counter = Counter()

    for dirpath, dirnames, filenames in os.walk(root):
        # Editing dirnames in place stops os.walk from descending into ignored folders.
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]

        for name in filenames:
            path = Path(dirpath) / name
            language = detect_language(path)
            if name in IGNORED_FILENAMES:
                skipped["lock/generated file"] += 1
                continue
            if language is None:
                skipped["not a code file"] += 1
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                skipped["too large"] += 1
                continue

            data = path.read_bytes()
            if b"\0" in data[:8000]:  # a null byte means it's binary, not text
                skipped["binary"] += 1
                continue

            text = data.decode("utf-8", errors="replace")
            files.append(SourceFile(
                path=path.relative_to(root).as_posix(),
                language=language,
                text=text,
                lines=text.count("\n") + (0 if text.endswith("\n") or not text else 1),
            ))

    files.sort(key=lambda f: f.path)
    return files, skipped


def print_summary(root: Path, files: list[SourceFile], skipped: Counter) -> None:
    total_lines = sum(f.lines for f in files)
    print(f"\nRepo: {root}")
    print(f"Kept {len(files)} files, {total_lines:,} lines")

    print("\nBy language:")
    lines_by_lang: Counter = Counter()
    for f in files:
        lines_by_lang[f.language] += f.lines
    for lang, lines in lines_by_lang.most_common():
        print(f"  {lang:<18} {lines:>8,} lines")

    print("\nLargest files:")
    for f in sorted(files, key=lambda f: f.lines, reverse=True)[:5]:
        print(f"  {f.lines:>6,}  {f.path}")

    if skipped:
        print("\nSkipped:")
        for reason, count in skipped.most_common():
            print(f"  {count:>5}  {reason}")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    root = resolve_source(sys.argv[1])
    files, skipped = collect_files(root)
    print_summary(root, files, skipped)


if __name__ == "__main__":
    main()
