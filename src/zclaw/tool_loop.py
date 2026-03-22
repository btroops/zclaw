"""Two-stage flow: model emits tool JSON → execute tools → model synthesizes reply."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from zclaw.prompts import SECOND_STAGE_PROMPT, TOOL_CALL_PROMPT_OPTIMIZED
from zclaw.tools import (
    TOOL_REGISTRY,
    discover_directories_by_name,
    discover_files_by_basename,
)


def _clip_for_context(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n… （内容过长，已截断以适配模型上下文）"


def _llm_with_max_tokens(llm: BaseChatModel, max_tokens: int) -> BaseChatModel:
    if hasattr(llm, "model_copy"):
        try:
            return llm.model_copy(update={"max_tokens": max_tokens})
        except Exception:
            pass
    try:
        llm.max_tokens = max_tokens  # type: ignore[attr-defined]
    except Exception:
        pass
    return llm


def build_tool_call_prompt(*, default_root_dir: str, user_instruction: str) -> str:
    base = TOOL_CALL_PROMPT_OPTIMIZED.format(
        default_root_dir=default_root_dir,
        user_instruction=user_instruction,
    )
    # 与 prompts 中「路径规则」呼应，避免模型仍强迫用户给绝对路径
    return (
        base
        + "\n### 再次强调（路径）\n"
        + "用户只说「某某目录」「某某文件夹」且未写完整路径时，在 JSON 里把 `root_dir` 或 `file_path` 写成"
        + " **相对默认工程根目录** 的路径即可（例如 `llm`、`src/zclaw`、`README.md`）。"
        + "程序解析顺序为：先默认工程根目录，再当前工作目录（若工程根下不存在同名路径）。\n"
    )


def extract_json_object(text: str) -> str:
    """Strip fences and isolate the first top-level JSON object in ``text``."""
    s = text.strip()
    # 只认 ```json，避免 ```python 等被 (?:json)? 误匹配成空并吞掉全文
    m = re.search(r"```json\s*([\s\S]*?)```", s, re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    start = s.find("{")
    if start == -1:
        return s
    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(s[start:])
        return s[start : start + end]
    except json.JSONDecodeError:
        end = s.rfind("}")
        if end > start:
            return s[start : end + 1]
        return s[start:]


def parse_tool_call(text: str) -> Dict[str, Any]:
    raw = extract_json_object(text)
    return json.loads(raw)


def resolve_path_with_meta(user_path: str, default_root: str) -> Tuple[str, bool]:
    """
    Resolve path for tools. Returns ``(resolved, used_cwd_fallback)``.
    Relative paths: try session workspace first, then current working directory.
    """
    p = user_path.strip()
    if not p:
        return default_root, False
    from pathlib import Path

    exp = Path(p).expanduser()
    if exp.is_absolute():
        return str(exp.resolve()), False

    rel = exp
    root = Path(default_root).resolve()
    cwd = Path.cwd().resolve()

    under_workspace = (root / rel).resolve()
    if under_workspace.exists():
        return str(under_workspace), False

    under_cwd = (cwd / rel).resolve()
    if under_cwd.exists():
        return str(under_cwd), True

    return str(under_workspace), False


def resolve_path(user_path: str, default_root: str) -> str:
    """Same as :func:`resolve_path_with_meta` but only returns the path string."""
    return resolve_path_with_meta(user_path, default_root)[0]


def _is_simple_filename_for_search(path_str: str) -> bool:
    """No path separators → allow basename search under workspace (e.g. ``tools.py``)."""
    r = path_str.strip().replace("\\", "/")
    return bool(r) and "/" not in r


def execute_tool_call(
    parsed: Dict[str, Any],
    *,
    default_root_dir: str,
) -> Tuple[Optional[str], str]:
    """
    Run at most one tool from parsed JSON.
    Returns (tool_name or None, result text for the model).
    """
    name = parsed.get("tool_name")
    params = parsed.get("tool_params") or {}

    if name is None or name == "null":
        return None, "（未调用工具：模型判定无需读取本地目录或文件。）"

    if name not in TOOL_REGISTRY:
        return None, f"错误：未知工具 {name!r}，未执行任何本地工具。"

    fn = TOOL_REGISTRY[name]
    try:
        if name == "get_project_directory":
            raw_root = params.get("root_dir")
            root_dir, cwd_fb = resolve_path_with_meta(
                str(raw_root).strip() if raw_root is not None else "",
                default_root_dir,
            )
            discover_note = ""
            if not Path(root_dir).is_dir():
                raw = str(raw_root).strip() if raw_root is not None else ""
                exp = Path(raw).expanduser() if raw else Path()
                if raw and not exp.is_absolute():
                    needle = exp.name if exp.name else raw.rstrip("/").split("/")[-1]
                    if needle and needle not in (".", ".."):
                        hits_ws = discover_directories_by_name(
                            default_root_dir, needle, max_depth=16, max_results=25
                        )
                        if len(hits_ws) == 1:
                            root_dir = hits_ws[0]
                            discover_note = (
                                "（路径说明：工程根下无直接路径，已按文件夹名在**默认工程根**内"
                                "检索到唯一匹配目录。）\n\n"
                            )
                        elif len(hits_ws) > 1:
                            lines = "\n".join(f"- {h}" for h in hits_ws[:20])
                            return name, (
                                f"未找到唯一目录「{needle}」（直接路径不存在）。"
                                f"在默认工程根下检索到多个同名或相近文件夹，请指定完整路径或其一：\n{lines}"
                            )
                        else:
                            hits_cwd: list[str] = []
                            if Path.cwd().resolve() != Path(default_root_dir).resolve():
                                hits_cwd = discover_directories_by_name(
                                    str(Path.cwd()), needle, max_depth=16, max_results=25
                                )
                            if len(hits_cwd) == 1:
                                root_dir = hits_cwd[0]
                                discover_note = (
                                    "（路径说明：默认工程根下无此路径且未检索到同名文件夹，"
                                    "已在**当前工作目录**树下检索到唯一匹配。）\n\n"
                                )
                            elif len(hits_cwd) > 1:
                                lines = "\n".join(f"- {h}" for h in hits_cwd[:20])
                                return name, (
                                    f"未找到唯一目录「{needle}」。"
                                    f"工程根下无匹配，当前工作目录树下有多个候选，请指定其一：\n{lines}"
                                )

            exclude = params.get("exclude_dirs")
            md = int(params.get("max_depth", 5))
            out = fn(
                root_dir,
                exclude_dirs=exclude,
                max_depth=md,
            )
            if discover_note:
                out = discover_note + out
            if cwd_fb:
                out = (
                    "（路径说明：相对路径在默认工程根下未找到，已使用当前工作目录下的匹配路径。）\n\n"
                    + out
                )
            return name, out
        if name == "get_file_content":
            fp = params.get("file_path")
            if fp is None or not str(fp).strip():
                return name, "错误：未提供 file_path（文件绝对路径或相对默认工程根目录的路径）。"
            raw_fp = str(fp).strip()
            fp, cwd_fb = resolve_path_with_meta(raw_fp, default_root_dir)
            discover_note = ""
            if not Path(fp).is_file() and _is_simple_filename_for_search(raw_fp):
                exp = Path(raw_fp).expanduser()
                if not exp.is_absolute():
                    base = Path(raw_fp).name
                    if base:
                        hits_ws = discover_files_by_basename(
                            default_root_dir, base, max_depth=16, max_results=25
                        )
                        if len(hits_ws) == 1:
                            fp = hits_ws[0]
                            discover_note = (
                                "（路径说明：工程根下无此直接路径，已按**文件名**在默认工程根内"
                                "检索到唯一匹配。）\n\n"
                            )
                        elif len(hits_ws) > 1:
                            lines = "\n".join(f"- {h}" for h in hits_ws[:20])
                            return name, (
                                f"未找到唯一文件「{base}」。在默认工程根下有多份同名文件，请指定路径：\n{lines}"
                            )
                        else:
                            hits_cwd: list[str] = []
                            if Path.cwd().resolve() != Path(default_root_dir).resolve():
                                hits_cwd = discover_files_by_basename(
                                    str(Path.cwd()), base, max_depth=16, max_results=25
                                )
                            if len(hits_cwd) == 1:
                                fp = hits_cwd[0]
                                discover_note = (
                                    "（路径说明：工程根下无此文件，已在当前工作目录树下"
                                    "按文件名检索到唯一匹配。）\n\n"
                                )
                            elif len(hits_cwd) > 1:
                                lines = "\n".join(f"- {h}" for h in hits_cwd[:20])
                                return name, (
                                    f"工程根下无「{base}」，当前目录树下多个同名文件，请指定其一：\n{lines}"
                                )
            out = fn(fp)
            if discover_note:
                out = discover_note + out
            if cwd_fb:
                out = (
                    "（路径说明：相对路径在默认工程根下未找到，已使用当前工作目录下的匹配路径。）\n\n"
                    + out
                )
            return name, out
    except Exception as e:  # noqa: BLE001 — surface to model
        return name, f"错误：执行工具 {name} 时异常：{e}"

    return None, f"错误：无法为工具 {name!r} 解析参数。"


@dataclass
class ToolLoopResult:
    tool_call_raw: str
    tool_call_parsed: Dict[str, Any]
    tool_name: Optional[str]
    tool_output: str
    final_reply: str


def run_tool_loop(
    llm: BaseChatModel,
    user_instruction: str,
    *,
    default_root_dir: str,
    stage1_max_tokens: int = 512,
    stage2_max_tokens: int = 512,
) -> ToolLoopResult:
    """
    1) System prompt with TOOL_CALL_PROMPT_OPTIMIZED → model outputs JSON only (best effort).
    2) Execute tool locally.
    3) Second message with SECOND_STAGE_PROMPT → final natural language reply.
    """
    stage1_system = build_tool_call_prompt(
        default_root_dir=default_root_dir,
        user_instruction=user_instruction,
    )
    llm_s1 = _llm_with_max_tokens(llm, stage1_max_tokens)

    m1 = llm_s1.invoke(
        [
            SystemMessage(
                content=stage1_system
                + "\n\n【重要】请只输出一条合法 JSON 对象，不要输出任何其它文字或 Markdown。"
                + "禁止在 JSON 之后追加标题、说明、代码块或「生成回复」等第二段内容。"
            ),
        ]
    )
    raw = m1.content if isinstance(m1, AIMessage) else str(m1)
    tool_call_raw = raw if isinstance(raw, str) else str(raw)

    tool_name: Optional[str] = None
    try:
        parsed = parse_tool_call(tool_call_raw)
    except (json.JSONDecodeError, ValueError) as e:
        parsed = {
            "tool_name": None,
            "tool_params": {},
            "reason": f"JSON 解析失败：{e}",
        }
        tool_output = (
            f"（工具调用 JSON 无法解析，请根据用户指令直接回复。原始输出：\n{tool_call_raw[:2000]}\n）"
        )
    else:
        tool_name, tool_output = execute_tool_call(parsed, default_root_dir=default_root_dir)

    llm_s2 = _llm_with_max_tokens(llm, stage2_max_tokens)

    out_for_model = _clip_for_context(tool_output, 2000)
    stage2_user = (
        f"用户指令：\n{user_instruction}\n\n"
        f"模型输出的工具调用 JSON（原文）：\n{tool_call_raw[:4000]}\n\n"
        f"工具执行结果（供你生成最终回复，须基于真实数据）：\n{out_for_model}\n"
    )
    m2 = llm_s2.invoke(
        [
            SystemMessage(content=SECOND_STAGE_PROMPT),
            HumanMessage(content=stage2_user),
        ]
    )
    final = m2.content if isinstance(m2, AIMessage) else str(m2)
    final_reply = final if isinstance(final, str) else str(final)

    return ToolLoopResult(
        tool_call_raw=tool_call_raw,
        tool_call_parsed=parsed,
        tool_name=tool_name,
        tool_output=tool_output,
        final_reply=final_reply,
    )
