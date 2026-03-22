"""
从自然语言指令中推断「在某父目录下创建/删除子项」的相对路径，用于校正模型漏写的父路径段。

仅做保守规则匹配（正则），不调用 LLM；仅在模型给出的路径为**单层且与解析出的子项同名**时改写为 parent/child。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# 在 src 目录下 / 在 src 目录里 → 创建 functest 目录
_UNDER_DIR_CREATE_DIR = re.compile(
    r"在\s*([^\s/]+(?:/[^\s/]+)*)\s*目录(?:里|下)?\s*创建\s*(?:一个|名为)?\s*([^\s/]+)\s*(?:目录|文件夹)?",
    re.IGNORECASE,
)
# 在 src 下 → 创建 functest 目录（父级后无「目录」二字）
_UNDER_CREATE_DIR = re.compile(
    r"在\s*([^\s/]+(?:/[^\s/]+)*)\s*下\s*创建\s*(?:一个|名为)?\s*([^\s/]+)\s*(?:目录|文件夹)?",
    re.IGNORECASE,
)
_UNDER_DIR_CREATE_FILE = re.compile(
    r"在\s*([^\s/]+(?:/[^\s/]+)*)\s*目录(?:里|下)?\s*创建\s*(?:一个|名为)?\s*([^\s/]+)\s*文件",
    re.IGNORECASE,
)
_UNDER_CREATE_FILE = re.compile(
    r"在\s*([^\s/]+(?:/[^\s/]+)*)\s*下\s*创建\s*(?:一个|名为)?\s*([^\s/]+)\s*文件",
    re.IGNORECASE,
)
# 「在 xxx 目录下创建 Hello.java」——无「文件」二字，但扩展名表明是文件
_UNDER_DIR_CREATE_FILE_LOOSE = re.compile(
    r"在\s*([^\s/]+(?:/[^\s/]+)*)\s*目录(?:里|下)?\s*创建\s*(?:一个|名为)?\s*([^\s/]+\.[^\s/.]+)\s*",
    re.IGNORECASE,
)
_UNDER_CREATE_FILE_LOOSE = re.compile(
    r"在\s*([^\s/]+(?:/[^\s/]+)*)\s*下\s*创建\s*(?:一个|名为)?\s*([^\s/]+\.[^\s/.]+)\s*",
    re.IGNORECASE,
)
# 「在 src/functest 创建 HelloWorld.java」——无「下」「目录」，直接 路径 + 创建 + 带后缀文件名
_UNDER_INLINE_CREATE_FILE = re.compile(
    r"在\s*([^\s/]+(?:/[^\s/]+)*)\s*创建\s*(?:一个|名为)?\s*([^\s/]+\.[^\s/.]+)\s*",
    re.IGNORECASE,
)
# 向/在 src 目录下 写入 notes.txt（捕获父路径与文件名）
_UNDER_DIR_WRITE_FILE = re.compile(
    r"(?:向|在)\s*([^\s/]+(?:/[^\s/]+)*)\s*目录(?:里|下)?\s*(?:写入|追加|保存)\s*([^\s/]+)",
    re.IGNORECASE,
)
_UNDER_WRITE_FILE = re.compile(
    r"(?:向|在)\s*([^\s/]+(?:/[^\s/]+)*)\s*下\s*(?:写入|追加|保存)\s*([^\s/]+)",
    re.IGNORECASE,
)
# 仅有「在 xxx 下 写入」且文件名在 tool_params 里
_UNDER_DIR_WRITE = re.compile(
    r"(?:向|在)\s*([^\s/]+(?:/[^\s/]+)*)\s*目录(?:里|下)?\s*(?:写入|追加|保存)\s*$",
    re.IGNORECASE,
)
_UNDER_WRITE = re.compile(
    r"(?:向|在)\s*([^\s/]+(?:/[^\s/]+)*)\s*下\s*(?:写入|追加|保存)\s*$",
    re.IGNORECASE,
)
# 在 src 目录(里/下/直接接)删除 functest 目录
_UNDER_DIR_DELETE_DIR = re.compile(
    r"在\s*([^\s/]+(?:/[^\s/]+)*)\s*目录(?:里|下)?\s*删除\s*(?:一个|名为)?\s*([^\s/]+)\s*(?:目录|文件夹)?",
    re.IGNORECASE,
)
_UNDER_DELETE_DIR = re.compile(
    r"在\s*([^\s/]+(?:/[^\s/]+)*)\s*下\s*删除\s*(?:一个|名为)?\s*([^\s/]+)\s*(?:目录|文件夹)?",
    re.IGNORECASE,
)


def _norm_seg(s: str) -> str:
    return s.strip().replace("\\", "/").strip("/")


# 常见源码/文本后缀：用于区分「创建目录」与「创建文件」、以及 create_directory 误用纠正
_KNOWN_FILE_SUFFIXES = frozenset(
    {
        ".java",
        ".py",
        ".pyi",
        ".pyw",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cc",
        ".cxx",
        ".go",
        ".rs",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".kt",
        ".scala",
        ".php",
        ".rb",
        ".swift",
        ".txt",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".xml",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".less",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".bat",
        ".cmd",
        ".ps1",
        ".properties",
        ".gradle",
        ".groovy",
        ".vue",
        ".svelte",
        ".cs",
        ".fs",
        ".vb",
        ".r",
        ".m",
        ".mm",
        ".pl",
        ".pm",
        ".lua",
        ".dart",
        ".ex",
        ".exs",
        ".erl",
        ".hs",
        ".clj",
        ".cljs",
        ".coffee",
        ".tex",
        ".rst",
        ".adoc",
        ".csv",
        ".tsv",
        ".log",
        ".env",
        ".gitignore",
        ".dockerignore",
    }
)


def path_last_segment_looks_like_file(path_or_name: str) -> bool:
    """路径最后一段是否像文件（基于后缀白名单）。"""
    name = path_or_name.strip().replace("\\", "/").split("/")[-1]
    if not name or name.startswith("."):
        return False
    suf = Path(name).suffix.lower()
    return suf in _KNOWN_FILE_SUFFIXES


def maybe_coerce_directory_to_create_file(
    tool_name: str,
    params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], str]:
    """
    若模型误用 ``create_directory`` 但 ``dir_path`` 实为文件路径（如 ``a/b/Hello.java``），
    改为 ``create_file``。返回 (tool_name, params, 说明前缀)。
    """
    if tool_name != "create_directory":
        return tool_name, params, ""
    dp = params.get("dir_path")
    if dp is None or not str(dp).strip():
        return tool_name, params, ""
    raw = str(dp).strip().replace("\\", "/")
    if not path_last_segment_looks_like_file(raw):
        return tool_name, params, ""
    new_params = dict(params)
    new_params.pop("dir_path", None)
    new_params.pop("parents", None)
    new_params.pop("exist_ok", None)
    new_params["file_path"] = raw
    new_params.setdefault("content", "")
    note = (
        "（工具校正：路径以常见文件后缀结尾，已将 create_directory 改为 create_file。）\n\n"
    )
    return "create_file", new_params, note


def maybe_coerce_create_directory_to_create_file_from_user_intent(
    user_instruction: Optional[str],
    tool_name: str,
    params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], str]:
    """
    用户话里明确是「在某目录下创建带扩展名的文件」，但模型误选 ``create_directory``
    （且 ``dir_path`` 往往只写到父目录）时，改为 ``create_file``。
    """
    if not user_instruction or not str(user_instruction).strip():
        return tool_name, params, ""
    if tool_name != "create_directory":
        return tool_name, params, ""
    ext = extract_under_create_file(str(user_instruction).strip())
    if not ext:
        return tool_name, params, ""
    parent, fname = ext
    if not path_last_segment_looks_like_file(fname):
        return tool_name, params, ""
    intended = _intended_path(parent, fname)
    new_params = dict(params)
    new_params.pop("dir_path", None)
    new_params.pop("parents", None)
    new_params.pop("exist_ok", None)
    new_params["file_path"] = intended
    new_params.setdefault("content", "")
    note = (
        "（工具校正：用户指令为在目录下创建带扩展名的文件，已将 create_directory 改为 create_file，"
        f"file_path 为「{intended}」。）\n\n"
    )
    return "create_file", new_params, note


# 「在 path/to/File.java 中写/改…」→ 提取路径，用于纠正模型拼错的文件名
_IN_FILE_EDIT_PATH = re.compile(
    r"在\s*([^\s/]+(?:/[^\s/]+)*\.(?:java|cs|cpp|cc|cxx|c|h|hpp|hh|py|pyi|js|mjs|cjs|ts|tsx|jsx|go|rs|kt|kts|txt|md|json|sql|sh|gradle|xml|yaml|yml))\s*中",
    re.IGNORECASE,
)


def extract_in_file_path_from_instruction(text: str) -> Optional[str]:
    """从「在 xxx.yyy 中…」提取相对路径（工程内文件）。"""
    m = _IN_FILE_EDIT_PATH.search(text.strip())
    if not m:
        return None
    return m.group(1).strip().replace("\\", "/")


def maybe_align_file_path_with_user_instruction(
    user_instruction: Optional[str],
    tool_name: str,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """
    当用户指令明确写出「在 … 文件 中」操作时，将 ``file_path`` 与用户写出路径对齐，
    避免模型把 ``HelloWold`` 写成 ``HelloWorld`` 等。
    """
    if not user_instruction or not str(user_instruction).strip():
        return params, ""
    if tool_name not in ("create_file", "write_file", "append_file"):
        return params, ""
    hint = extract_in_file_path_from_instruction(str(user_instruction).strip())
    if not hint:
        return params, ""
    fp = params.get("file_path")
    p = dict(params)
    if fp is None or not str(fp).strip():
        p["file_path"] = hint
        return p, f"（路径校正：根据用户指令「在…中」，已设 file_path 为「{hint}」。）\n\n"
    cur = str(fp).strip().replace("\\", "/")
    if cur.casefold() == hint.casefold():
        return p, ""
    p["file_path"] = hint
    return (
        p,
        f"（路径校正：用户指令中的文件为「{hint}」，已将 file_path 由「{cur}」对齐。）\n\n",
    )


def _clean_leaf_name(name: str, *, kind: str = "any") -> str:
    """
    去掉模型/正则多捕获的「xxx目录」「xxx文件」等后缀，得到纯目录名或文件名。
    kind: 'dir' | 'file' | 'any'
    """
    n = name.strip()
    suffixes: Tuple[str, ...]
    if kind == "dir":
        suffixes = ("目录", "文件夹")
    elif kind == "file":
        suffixes = ("文件",)
    else:
        suffixes = ("目录", "文件夹", "文件")
    changed = True
    while changed and n:
        changed = False
        for suf in suffixes:
            if n.endswith(suf) and len(n) > len(suf):
                n = n[: -len(suf)].strip()
                changed = True
                break
    return n


def extract_under_delete_directory(text: str) -> Optional[Tuple[str, str]]:
    """若匹配「在父路径下删除子目录」，返回 (parent_relpath, child_dirname)。"""
    t = text.strip()
    for rx in (_UNDER_DIR_DELETE_DIR, _UNDER_DELETE_DIR):
        m = rx.search(t)
        if m:
            parent, child = m.group(1), m.group(2)
            if parent and child:
                return _norm_seg(parent), _clean_leaf_name(child, kind="dir")
    return None


def extract_under_create_directory(text: str) -> Optional[Tuple[str, str]]:
    """若匹配「在父路径下创建子目录」，返回 (parent_relpath, child_dirname)。"""
    t = text.strip()
    for rx in (_UNDER_DIR_CREATE_DIR, _UNDER_CREATE_DIR):
        m = rx.search(t)
        if m:
            parent, child = m.group(1), m.group(2)
            if parent and child:
                child = _clean_leaf_name(child, kind="dir")
                if path_last_segment_looks_like_file(child):
                    return None
                return _norm_seg(parent), child
    return None


def extract_under_create_file(text: str) -> Optional[Tuple[str, str]]:
    """若匹配「在父路径下创建某文件」，返回 (parent_relpath, filename)。"""
    t = text.strip()
    for rx in (
        _UNDER_DIR_CREATE_FILE,
        _UNDER_CREATE_FILE,
        _UNDER_DIR_CREATE_FILE_LOOSE,
        _UNDER_CREATE_FILE_LOOSE,
        _UNDER_INLINE_CREATE_FILE,
    ):
        m = rx.search(t)
        if m:
            parent, fname = m.group(1), m.group(2)
            if parent and fname:
                return _norm_seg(parent), _clean_leaf_name(fname, kind="file")
    return None


def extract_under_write_file(text: str) -> Optional[Tuple[str, str]]:
    """「在 parent 下 写入 fname」→ (parent_relpath, filename)。"""
    t = text.strip()
    for rx in (_UNDER_DIR_WRITE_FILE, _UNDER_WRITE_FILE):
        m = rx.search(t)
        if m:
            parent, fname = m.group(1), m.group(2)
            if parent and fname:
                return _norm_seg(parent), fname.strip()
    return None


def extract_parent_for_write_path(text: str) -> Optional[str]:
    """仅父路径，用于指令里未写文件名、由模型给出单文件名的情况。"""
    t = text.strip()
    for rx in (_UNDER_DIR_WRITE, _UNDER_WRITE):
        m = rx.search(t)
        if m:
            parent = m.group(1)
            if parent:
                return _norm_seg(parent)
    return None


def _single_segment_matches_child(current: str, child: str) -> bool:
    cur = current.strip().replace("\\", "/")
    ch = child.strip().replace("\\", "/")
    if not ch or cur.casefold() != ch.casefold():
        return False
    return "/" not in cur


def _intended_path(parent: str, child: str) -> str:
    p, c = _norm_seg(parent), child.strip().replace("\\", "/")
    return f"{p}/{c}" if p else c


_MUTATION_TOOLS = frozenset(
    {
        "create_directory",
        "create_file",
        "write_file",
        "append_file",
        "delete_directory",
        "rename_file",
    }
)


def apply_mutation_path_intent(
    user_instruction: str,
    tool_name: str,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """
    根据用户中文表述补全多级相对路径。

    Returns
    -------
    (new_params, note_prefix)
        note_prefix 为空字符串表示未改写；否则为简短说明，供拼入工具返回。
    """
    if tool_name not in _MUTATION_TOOLS:
        return dict(params), ""

    p = dict(params)
    note = ""

    if tool_name == "create_directory":
        got = p.get("dir_path")
        if got is None or not str(got).strip():
            return p, ""
        cur = str(got).strip().replace("\\", "/")
        ext = extract_under_create_directory(user_instruction)
        if not ext:
            return p, ""
        parent, child = ext
        intended = _intended_path(parent, child)
        if _single_segment_matches_child(cur, child) and intended != cur:
            p["dir_path"] = intended
            note = (
                f"（路径校正：根据用户指令「在 {parent} 下创建 {child}」，"
                f"已将 dir_path 由「{cur}」改为「{intended}」。）\n\n"
            )
        return p, note

    if tool_name == "delete_directory":
        got = p.get("dir_path")
        if got is None or not str(got).strip():
            return p, ""
        cur = str(got).strip().replace("\\", "/")
        note = ""
        ext = extract_under_delete_directory(user_instruction)
        if not ext:
            return p, ""
        parent, child = ext
        intended = _intended_path(parent, child)
        if _single_segment_matches_child(cur, child) and intended != cur:
            p["dir_path"] = intended
            note = (
                f"（路径校正：根据用户指令「在 {parent} 下删除 {child}」，"
                f"已将 dir_path 由「{cur}」改为「{intended}」。）\n\n"
            )
        return p, note

    if tool_name in ("create_file", "write_file", "append_file"):
        got = p.get("file_path")
        if got is None or not str(got).strip():
            return p, ""
        cur = str(got).strip().replace("\\", "/")
        note = ""
        parent: Optional[str] = None
        fname: Optional[str] = None

        ext = extract_under_create_file(user_instruction)
        if ext:
            parent, fname = ext
        if parent is None:
            uw = extract_under_write_file(user_instruction)
            if uw:
                parent, fname = uw
        if parent is None or fname is None:
            parent_only = extract_parent_for_write_path(user_instruction)
            if parent_only:
                parent = parent_only
                base = cur.split("/")[-1]
                if base and _single_segment_matches_child(cur, base):
                    fname = base

        if parent and fname:
            intended = _intended_path(parent, fname)
            if _single_segment_matches_child(cur, fname) and intended != cur:
                p["file_path"] = intended
                note = (
                    f"（路径校正：根据用户指令中的父目录「{parent}」，"
                    f"已将 file_path 由「{cur}」改为「{intended}」。）\n\n"
                )
        return p, note

    if tool_name == "rename_file":
        op = p.get("old_path")
        np = p.get("new_path")
        if op is None or np is None:
            return p, ""
        note = ""
        n = str(np).strip().replace("\\", "/")
        ext = extract_under_create_directory(user_instruction)
        if ext:
            parent, child = ext
            intended = _intended_path(parent, child)
            # 仅当 new_path 被写成单层子目录名且与解析一致时，补全为 父/子
            if _single_segment_matches_child(n, child) and intended != n:
                p["new_path"] = intended
                note = (
                    f"（路径校正：根据用户指令「在 {parent} 下创建 {child}」，"
                    f"已将 new_path 由「{n}」改为「{intended}」。）\n\n"
                )
                return p, note
        # 目标为「仅文件名」且指令含「放到 parent 下」类语义
        extf = extract_under_create_file(user_instruction)
        if extf:
            parent, fname = extf
            intended = _intended_path(parent, fname)
            if _single_segment_matches_child(n, fname) and intended != n:
                p["new_path"] = intended
                note = (
                    f"（路径校正：根据用户指令，已将 new_path 由「{n}」改为「{intended}」。）\n\n"
                )
        return p, note

    return p, ""
