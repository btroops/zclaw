"""
在「会话默认工程根」下解析相对路径：先直配，再按目录名/文件名检索；必要时回退到当前工作目录。

默认工程根由 CLI 的 ``--root`` / 环境变量 ``ZCLAW_WORKSPACE`` / ``cwd`` 决定，不是操作系统根目录。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from zclaw.tools import discover_directories_by_name, discover_files_by_basename


def _split_segments(raw: str) -> List[str]:
    r = raw.strip().replace("\\", "/")
    return [p for p in r.split("/") if p and p != "."]


def _resolve_directory_chain_from(
    base: Path,
    workspace_root: Path,
    segments: List[str],
) -> Tuple[Optional[Path], str]:
    """
    Walk ``segments`` as directory names under ``base``.
    Each step: literal ``base/seg``, else name search under ``base``, else name search under ``workspace_root`` (re-anchor).
    Returns (resolved directory, note) or (None, error_message).
    """
    if not segments:
        return base.resolve(), ""

    current = base.resolve()
    note_parts: list[str] = []

    for seg in segments:
        direct = (current / seg).resolve()
        if direct.is_dir():
            current = direct
            continue

        hits_local = discover_directories_by_name(str(current), seg, max_depth=16, max_results=25)
        if len(hits_local) == 1:
            current = Path(hits_local[0])
            note_parts.append(f"「{seg}」")
            continue
        if len(hits_local) > 1:
            lines = "\n".join(f"- {h}" for h in hits_local[:15])
            return None, (
                f"目录段「{seg}」在「{current}」下匹配到多个文件夹，请指定完整路径：\n{lines}"
            )

        hits_ws = discover_directories_by_name(str(workspace_root), seg, max_depth=16, max_results=25)
        if len(hits_ws) == 1:
            current = Path(hits_ws[0])
            note_parts.append(f"「{seg}」（在工程根内检索）")
            continue
        if len(hits_ws) > 1:
            lines = "\n".join(f"- {h}" for h in hits_ws[:15])
            return None, f"目录段「{seg}」在工程根下匹配到多个文件夹：\n{lines}"

        return None, (
            f"未找到目录段「{seg}」（自 {current} 起按名称检索无结果）。"
        )

    detail = "、".join(note_parts) if note_parts else ""
    note = (
        f"（路径说明：部分目录段通过名称检索匹配：{detail}。）\n\n" if detail else ""
    )
    return current, note


def resolve_target_directory(
    workspace_root: str,
    raw: str,
    *,
    cwd_fallback: bool = True,
) -> Tuple[Optional[str], str]:
    """
    解析用户/模型给出的**目录**相对路径（可多级，如 ``llm``、``src/zclaw``）。

    顺序：工程根下直接路径 → 工程根下目录链检索 →（可选）当前工作目录同样流程。
    返回 ``(绝对路径, 说明前缀或错误信息)``；失败时路径为 ``None``，第二项为错误说明。
    """
    ws = Path(workspace_root).expanduser().resolve()
    if not ws.is_dir():
        return None, f"错误：会话默认工程根不是目录：{ws}"

    r = raw.strip() if raw else ""
    if not r:
        return str(ws), ""

    exp = Path(r).expanduser()
    if exp.is_absolute():
        p = exp.resolve()
        if not p.is_dir():
            return None, f"错误：绝对路径不是目录：{p}"
        return str(p), ""

    rel = Path(r)
    direct_ws = (ws / rel).resolve()
    if direct_ws.is_dir():
        return str(direct_ws), ""

    cwd = Path.cwd().resolve()
    if cwd_fallback and cwd != ws:
        direct_cwd = (cwd / rel).resolve()
        if direct_cwd.is_dir():
            return str(direct_cwd), "（路径说明：目录在**当前工作目录**下解析。）\n\n"

    segments = _split_segments(r)
    if not segments:
        return str(ws), ""

    path, err = _resolve_directory_chain_from(ws, ws, segments)
    if path is not None:
        return str(path), err

    if cwd_fallback and cwd != ws:
        path2, err2 = _resolve_directory_chain_from(cwd, ws, segments)
        if path2 is not None:
            return str(path2), "（路径说明：在工程根未解析，已在**当前工作目录**树下解析。）\n\n" + err2
        return None, err + "\n" + err2

    return None, err


def resolve_target_file(
    workspace_root: str,
    raw: str,
    *,
    cwd_fallback: bool = True,
) -> Tuple[Optional[str], str]:
    """
    解析**文件**路径。

    - 仅文件名（无 ``/``）：工程根直配 → cwd 直配 → 工程根内按文件名检索 → cwd 内按文件名检索。
    - 多级（如 ``llm/tools.py``）：先按目录链解析父路径，再 ``父路径/文件名``；若无直配则在父目录子树内按文件名检索。
    """
    ws = Path(workspace_root).expanduser().resolve()
    r = raw.strip()
    if not r:
        return None, "错误：空文件路径"

    exp = Path(r).expanduser()
    if exp.is_absolute():
        p = exp.resolve()
        if not p.is_file():
            return None, f"错误：不是文件或不存在：{p}"
        return str(p), ""

    segments = _split_segments(r)
    if len(segments) == 1:
        fname = segments[0]
        for base, label in ((ws, "工程根"), (Path.cwd().resolve(), "当前工作目录")):
            if base != Path.cwd().resolve() or cwd_fallback:
                direct = (base / fname).resolve()
                if direct.is_file():
                    note = "" if base == ws else f"（路径说明：文件在**{label}**下解析。）\n\n"
                    return str(direct), note
        hits_ws = discover_files_by_basename(str(ws), fname, max_depth=16, max_results=25)
        if len(hits_ws) == 1:
            return hits_ws[0], (
                "（路径说明：工程根下无直接路径，已按**文件名**在默认工程根内检索到唯一匹配。）\n\n"
            )
        if len(hits_ws) > 1:
            lines = "\n".join(f"- {h}" for h in hits_ws[:20])
            return None, f"未找到唯一文件「{fname}」，工程根下有多份同名文件：\n{lines}"

        if cwd_fallback and Path.cwd().resolve() != ws:
            hits_c = discover_files_by_basename(str(Path.cwd()), fname, max_depth=16, max_results=25)
            if len(hits_c) == 1:
                return hits_c[0], (
                    "（路径说明：工程根下无此文件，已在**当前工作目录**树下按文件名检索到唯一匹配。）\n\n"
                )
            if len(hits_c) > 1:
                lines = "\n".join(f"- {h}" for h in hits_c[:20])
                return None, f"工程根下无「{fname}」，当前目录树下多个同名文件：\n{lines}"

        return None, f"错误：未找到文件「{fname}」（已检索默认工程根与当前工作目录）。"

    dir_segments = segments[:-1]
    fname = segments[-1]
    dir_raw = "/".join(dir_segments)

    parent, dir_note = resolve_target_directory(workspace_root, dir_raw, cwd_fallback=cwd_fallback)
    if parent is None:
        return None, dir_note

    parent_p = Path(parent)
    direct_f = (parent_p / fname).resolve()
    if direct_f.is_file():
        return str(direct_f), dir_note

    hits = discover_files_by_basename(str(parent_p), fname, max_depth=16, max_results=25)
    if len(hits) == 1:
        return hits[0], dir_note + (
            f"（路径说明：在目录「{parent_p.name}」下按文件名「{fname}」检索到唯一匹配。）\n\n"
        )
    if len(hits) > 1:
        lines = "\n".join(f"- {h}" for h in hits[:15])
        return None, (
            dir_note
            + f"在「{parent}」下有多份「{fname}」，请指定完整相对路径：\n{lines}"
        )

    return None, (
        dir_note + f"错误：在「{parent}」下未找到文件「{fname}」。"
    )
