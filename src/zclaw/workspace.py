"""Resolve project root and build a text digest so the model can \"see\" the workspace."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# Names to skip when walking the tree (common VCS, caches, venvs).
_IGNORE_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "node_modules",
        ".ipynb_checkpoints",
        ".tox",
        "dist",
        "build",
        ".eggs",
    }
)


def resolve_workspace_root(explicit: Optional[os.PathLike[str] | str] = None) -> Path:
    """
    Prefer explicit path, then env ``ZCLAW_WORKSPACE``, then current working directory.
    The path is resolved to an absolute, existing directory.
    """
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
    else:
        env = os.environ.get("ZCLAW_WORKSPACE")
        root = Path(env).expanduser().resolve() if env else Path.cwd().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Workspace root is not a directory: {root}")
    return root


def _read_readme_snippet(root: Path, max_chars: int) -> str:
    for name in ("README.md", "README.rst", "README.txt", "README"):
        p = root / name
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            text = text.strip()
            if len(text) > max_chars:
                return text[:max_chars] + "\n… (truncated)"
            return text
    return "(no README found at project root)"


def _format_tree(
    root: Path,
    *,
    max_depth: int,
    max_lines: int,
) -> str:
    lines: list[str] = []
    count = 0

    def walk(current: Path, prefix: str, depth: int) -> None:
        nonlocal count
        if count >= max_lines or depth > max_depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            lines.append(f"{prefix}[inaccessible]")
            count += 1
            return

        for p in entries:
            if count >= max_lines:
                return
            name = p.name
            if name.startswith("."):
                continue
            if p.is_dir() and (
                name in _IGNORE_DIR_NAMES or name.endswith(".egg-info")
            ):
                continue

            lines.append(f"{prefix}{name}/" if p.is_dir() else f"{prefix}{name}")
            count += 1
            if p.is_dir() and depth < max_depth:
                walk(p, prefix + "  ", depth + 1)

    walk(root, "", 0)
    if count >= max_lines:
        lines.append("… (tree truncated)")
    return "\n".join(lines) if lines else "(empty)"


def build_workspace_digest(
    root: Optional[os.PathLike[str] | str] = None,
    *,
    max_tree_depth: int = 2,
    max_tree_lines: int = 120,
    readme_max_chars: int = 4000,
) -> str:
    """
    Human-readable summary: absolute root, OS cwd, shallow directory tree, README excerpt.
    Intended to be injected into a system prompt so the model knows \"where\" it is.
    """
    path = resolve_workspace_root(root)
    readme = _read_readme_snippet(path, readme_max_chars)
    tree = _format_tree(path, max_depth=max_tree_depth, max_lines=max_tree_lines)

    parts = [
        f"工作区根目录（绝对路径）: {path}",
        f"进程当前工作目录: {Path.cwd().resolve()}",
        "",
        "【工程目录树（深度与行数有限制）】",
        tree,
        "",
        "【项目根 README 摘要】",
        readme,
    ]
    return "\n".join(parts)


def build_system_prompt_with_workspace(
    base_instruction: str,
    root: Optional[os.PathLike[str] | str] = None,
    **digest_kwargs: Any,
) -> str:
    """Combine your base system instruction with ``build_workspace_digest``."""
    digest = build_workspace_digest(root, **digest_kwargs)
    return (
        f"{base_instruction.strip()}\n\n"
        "以下是当前会话关联的工程目录信息。回答与文件、路径、项目结构相关的问题时请以此为依据。\n\n"
        f"{digest}"
    )
