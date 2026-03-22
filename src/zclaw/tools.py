"""Local tools: directory tree, file read (UTF-8), and workspace-scoped mutations."""

from __future__ import annotations

import shutil
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


def discover_directories_by_name(
    workspace_root: str,
    needle: str,
    *,
    max_depth: int = 16,
    max_results: int = 25,
) -> List[str]:
    """
    Walk ``workspace_root`` (depth-limited) and return absolute paths of directories
    whose name matches ``needle`` case-insensitively (exact name, then substring).
    Skips the same hidden / cache directories as the tree tool.
    """
    root = Path(workspace_root).expanduser().resolve()
    if not root.is_dir():
        return []
    n = needle.strip()
    if not n or n in (".", ".."):
        return []
    nf = n.casefold()
    exclude = _normalize_exclude(None)
    scored: List[tuple[int, int, str]] = []

    def score_dirname(dname: str) -> Optional[int]:
        dfc = dname.casefold()
        if dfc == nf:
            return 0
        if nf in dfc:
            return 1
        if len(dname) >= 2 and dfc in nf:
            return 2
        return None

    stack: List[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            for child in sorted(current.iterdir(), key=lambda p: p.name.lower()):
                if not child.is_dir():
                    continue
                if _should_skip_dir(child.name, exclude):
                    continue
                sc = score_dirname(child.name)
                if sc is not None:
                    resolved = str(child.resolve())
                    scored.append((sc, len(resolved), resolved))
                if depth < max_depth:
                    stack.append((child, depth + 1))
        except OSError:
            continue

    scored.sort(key=lambda t: (t[0], t[1]))
    out: List[str] = []
    seen: set[str] = set()
    for _, _, pth in scored:
        if pth in seen:
            continue
        seen.add(pth)
        out.append(pth)
        if len(out) >= max_results:
            break
    return out


def discover_files_by_basename(
    workspace_root: str,
    basename: str,
    *,
    max_depth: int = 16,
    max_results: int = 25,
) -> List[str]:
    """
    Walk under ``workspace_root`` and return absolute paths of **files** whose name
    equals ``basename`` (case-insensitive). Skips the same directories as the tree tool.
    """
    root = Path(workspace_root).expanduser().resolve()
    b = basename.strip()
    if not root.is_dir() or not b:
        return []
    bcf = b.casefold()
    exclude = _normalize_exclude(None)
    scored: List[tuple[int, int, str]] = []

    stack: List[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            for child in sorted(current.iterdir(), key=lambda p: p.name.lower()):
                if child.is_dir():
                    if _should_skip_dir(child.name, exclude):
                        continue
                    if depth < max_depth:
                        stack.append((child, depth + 1))
                elif child.is_file():
                    n = child.name
                    if n == b:
                        rp = str(child.resolve())
                        scored.append((0, len(rp), rp))
                    elif n.casefold() == bcf:
                        rp = str(child.resolve())
                        scored.append((1, len(rp), rp))
        except OSError:
            continue

    scored.sort(key=lambda t: (t[0], t[1]))
    out: List[str] = []
    seen: set[str] = set()
    for _, _, pth in scored:
        if pth in seen:
            continue
        seen.add(pth)
        out.append(pth)
        if len(out) >= max_results:
            break
    return out


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


def create_directory(
    dir_path: str,
    *,
    parents: bool = True,
    exist_ok: bool = True,
) -> str:
    """创建目录（默认递归创建父目录；exist_ok 为 True 时若已存在不报错）。"""
    p = Path(dir_path).expanduser().resolve()
    try:
        p.mkdir(parents=parents, exist_ok=exist_ok)
        return f"成功：已创建目录（或已存在）：{p}"
    except OSError as e:
        return f"错误：无法创建目录：{p} ({e})"


def create_file(file_path: str, content: str = "") -> str:
    """新建 UTF-8 文本文件；若已存在则报错不覆盖。"""
    p = Path(file_path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            if p.is_dir():
                return f"错误：路径已存在且为目录，无法创建同名文件：{p}"
            return f"错误：文件已存在，未覆盖：{p}"
        p.write_text(content, encoding="utf-8")
        return f"成功：已创建文件 {p}（{len(content)} 字符）"
    except OSError as e:
        return f"错误：无法创建文件：{p} ({e})"


def write_file(file_path: str, content: str) -> str:
    """写入 UTF-8 文本文件（覆盖已有内容；若父目录不存在则创建）。"""
    p = Path(file_path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"成功：已写入文件 {p}（{len(content)} 字符）"
    except OSError as e:
        return f"错误：无法写入文件：{p} ({e})"


def append_file(file_path: str, content: str) -> str:
    """向 UTF-8 文本文件末尾追加内容（文件不存在则创建）。"""
    p = Path(file_path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
        return f"成功：已追加到文件 {p}（{len(content)} 字符）"
    except OSError as e:
        return f"错误：无法追加文件：{p} ({e})"


def delete_file(file_path: str) -> str:
    """删除文件（仅文件，不删除目录）。"""
    p = Path(file_path).expanduser().resolve()
    if not p.is_file():
        return f"错误：不是文件或不存在：{p}"
    try:
        p.unlink()
        return f"成功：已删除文件 {p}"
    except OSError as e:
        return f"错误：无法删除文件：{p} ({e})"


def delete_directory(dir_path: str, *, recursive: bool = True) -> str:
    """
    删除目录。``recursive=True`` 时删除整棵子树（含非空目录）；为 ``False`` 时仅删除空目录。
    """
    p = Path(dir_path).expanduser().resolve()
    if not p.is_dir():
        return f"错误：不是目录或不存在：{p}"
    try:
        if recursive:
            shutil.rmtree(p)
        else:
            p.rmdir()
        return f"成功：已删除目录 {p}"
    except OSError as e:
        return f"错误：无法删除目录：{p} ({e})"


def rename_file(old_path: str, new_path: str) -> str:
    """重命名或移动文件（目标路径不可已存在）。"""
    old = Path(old_path).expanduser().resolve()
    new = Path(new_path).expanduser().resolve()
    if not old.is_file():
        return f"错误：源不是文件或不存在：{old}"
    if new.exists():
        return f"错误：目标路径已存在：{new}"
    try:
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        return f"成功：已将 {old} 重命名为 {new}"
    except OSError as e:
        return f"错误：无法重命名：{e}"


TOOL_REGISTRY: dict[str, Any] = {
    "get_project_directory": get_project_directory,
    "get_file_content": get_file_content,
    "create_directory": create_directory,
    "create_file": create_file,
    "write_file": write_file,
    "append_file": append_file,
    "delete_file": delete_file,
    "delete_directory": delete_directory,
    "rename_file": rename_file,
}
