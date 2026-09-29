"""Phase 2a - Chunk: split source files into small, meaningful pieces.

Why chunk at all? An embedding model can only read a few hundred words at a time,
and a search result is only useful if it points at a specific place in the code.
So instead of indexing whole files, we index functions, classes and sections.

- Python files are split along real boundaries (functions, classes, methods)
  using Python's own parser (the `ast` module).
- Everything else is split into overlapping windows of lines. (Later we can add
  smarter splitting for other languages with a parser like tree-sitter.)
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

from repolens.ingest import SourceFile

# A chunk longer than this is split again - long chunks make vague search results.
MAX_CHUNK_LINES = 80
# Fallback window size, and how many lines neighbouring windows share so that
# code sitting on a boundary is never cut in half without appearing whole somewhere.
WINDOW_LINES = 60
WINDOW_OVERLAP = 10


@dataclass
class Chunk:
    path: str
    start_line: int   # 1-based, inclusive - these become the file:line citations
    end_line: int
    kind: str         # "function", "class", "method", "module", or "section"
    name: str         # e.g. "Session.request", or "" for plain sections
    language: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def embedding_text(self) -> str:
        """What we actually feed the embedding model.

        Adding the file path and name gives the model context the code alone lacks:
        `def get(self)` means much more inside `src/requests/sessions.py`.
        """
        header = f"File: {self.path}\n"
        if self.name:
            header += f"{self.kind.capitalize()}: {self.name}\n"
        return header + self.text


def chunk_file(file: SourceFile) -> list[Chunk]:
    if file.language == "Python":
        try:
            return _chunk_python(file)
        except SyntaxError:
            pass  # e.g. Python 2 code - fall back to plain windows
    return _chunk_windows(file, 1, file.lines, kind="section", name="")


def _chunk_python(file: SourceFile) -> list[Chunk]:
    tree = ast.parse(file.text)
    lines = file.text.splitlines()
    chunks: list[Chunk] = []
    covered_until = 0  # last line already inside some chunk

    def start_of(node: ast.AST) -> int:
        # Decorators (@property, @app.route...) sit above the def and belong with it.
        decorators = getattr(node, "decorator_list", [])
        return min([node.lineno] + [d.lineno for d in decorators])

    def add_gap(until: int) -> None:
        """Module-level code between definitions: imports, constants, setup."""
        gap_start = covered_until + 1
        if until >= gap_start and any(l.strip() for l in lines[gap_start - 1:until]):
            chunks.extend(_chunk_windows(file, gap_start, until, kind="module", name=""))

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start, end = start_of(node), node.end_lineno
        add_gap(start - 1)

        is_class = isinstance(node, ast.ClassDef)
        if is_class and end - start + 1 > MAX_CHUNK_LINES:
            # Big class: one chunk for the class header/docstring, one per method.
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            header_end = (start_of(methods[0]) - 1) if methods else end
            chunks.extend(_chunk_windows(file, start, header_end, kind="class", name=node.name))
            for m in methods:
                chunks.extend(_chunk_windows(file, start_of(m), m.end_lineno,
                                             kind="method", name=f"{node.name}.{m.name}"))
        else:
            chunks.extend(_chunk_windows(file, start, end,
                                         kind="class" if is_class else "function", name=node.name))
        covered_until = end

    add_gap(len(lines))
    return chunks


def _chunk_windows(file: SourceFile, start: int, end: int, kind: str, name: str) -> list[Chunk]:
    """Cut lines start..end into pieces of at most MAX_CHUNK_LINES (one piece if it fits)."""
    lines = file.text.splitlines()
    size = MAX_CHUNK_LINES if kind != "section" else WINDOW_LINES
    chunks: list[Chunk] = []
    pos = start
    while pos <= end:
        stop = min(pos + size - 1, end)
        text = "\n".join(lines[pos - 1:stop])
        if text.strip():
            chunks.append(Chunk(file.path, pos, stop, kind, name, file.language, text))
        if stop == end:
            break
        pos = stop + 1 - WINDOW_OVERLAP
    return chunks


def chunk_files(files: list[SourceFile]) -> list[Chunk]:
    return [c for f in files for c in chunk_file(f)]
