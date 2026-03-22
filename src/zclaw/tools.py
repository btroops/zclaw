"""Local tools: directory tree and file read (UTF-8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

# Cap total lines to avoid huge trees in LLM context
_DEFAULT_MAX_LINES = 800


def _normalize_exclude(exclude_dirs: Optional[Sequence[str]]) -> frozenset[str]:
    base = [".git", "venv", "__pycache__"]
    if exclude_dirs:
        return frozenset(base) | frozenset(exclude_dirs)
    return frozenset(base)


def _should_skip_dir(name: str, exclude: frozenset[str]) -> bool:
    if name.startswith("."):
        return True
    if name in exclude:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def _format_tree_lines(
    root: Path,
    *,
    exclude: frozenset[str],
    max_depth: int,
    max_lines: int,
) -> List[str]:
    lines: List[str] = []
    count = 0

    def add_line(s: str) -> bool:
        nonlocal count
        if count >= max_lines:
            return False
        lines.append(s)
        count += 1
        return True

    if not add_line(f"工程目录：{root.resolve()}"):
        return lines

    def walk(dir_path: Path, prefix: str, depth: int) -> None:
        nonlocal count
        if depth > max_depth:
            return
        try:
            entries = sorted(
                dir_path.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except OSError as e:
            add_line(f"{prefix}└── [无法访问: {e}]")
            return

        entries = [
            p
            for p in entries
            if not (p.is_dir() and _should_skip_dir(p.name, exclude))
            and not (p.is_file() and p.name.startswith("."))
        ]

        for i, p in enumerate(entries):
            if count >= max_lines:
                return
            is_last = i == len(entries) - 1
            branch = "└── " if is_last else "├── "
            display = f"{p.name}/" if p.is_dir() else p.name
            add_line(f"{prefix}{branch}{display}")
            if p.is_dir() and depth < max_depth:
                ext = "    " if is_last else "│   "
                walk(p, prefix + ext, depth + 1)

    walk(root, "", 0)

    if count >= max_lines:
        lines.append("… （目录行数已达上限，已截断）")
    return lines


def get_project_directory(
    root_dir: str,
    exclude_dirs: Optional[List[str]] = None,
    max_depth: int = 5,
    max_lines: int = _DEFAULT_MAX_LINES,
) -> str:
    """
    遍历工程目录，返回结构化目录树文本（排除指定目录名；隐藏以 ``.`` 开头的条目）。
    ``root_dir`` 须为绝对路径（由调用方与模型约定）。
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        return f"错误：目录不存在或不是文件夹：{root}"

    exclude = _normalize_exclude(exclude_dirs)
    depth = max(0, int(max_depth))
    lines = _format_tree_lines(
        root,
        exclude=exclude,
        max_depth=depth,
        max_lines=max_lines,
    )
    return "\n".join(lines)


def get_file_content(file_path: str) -> str:
    """读取 UTF-8 文本文件完整内容。"""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        return f"错误：不是文件或路径不存在：{path}"
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        return f"错误：文件不是合法 UTF-8 文本：{path} ({e})"
    except OSError as e:
        return f"错误：无法读取文件：{path} ({e})"


TOOL_REGISTRY: dict[str, Any] = {
    "get_project_directory": get_project_directory,
    "get_file_content": get_file_content,
}
