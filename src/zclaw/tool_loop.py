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
from zclaw.path_resolve import resolve_target_directory, resolve_target_file
from zclaw.tools import TOOL_REGISTRY


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
        + "程序解析顺序为：先**会话默认工程根**（见下文路径），再当前工作目录；"
        + "多级路径会先逐级解析目录再定位文件。\n"
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
            raw = str(raw_root).strip() if raw_root is not None else ""
            root_dir, prefix = resolve_target_directory(
                default_root_dir, raw, cwd_fallback=True
            )
            if root_dir is None:
                return name, prefix
            exclude = params.get("exclude_dirs")
            md = int(params.get("max_depth", 5))
            out = fn(
                root_dir,
                exclude_dirs=exclude,
                max_depth=md,
            )
            return name, prefix + out
        if name == "get_file_content":
            fp = params.get("file_path")
            if fp is None or not str(fp).strip():
                return name, "错误：未提供 file_path（文件绝对路径或相对默认工程根目录的路径）。"
            raw_fp = str(fp).strip()
            path, prefix = resolve_target_file(
                default_root_dir, raw_fp, cwd_fallback=True
            )
            if path is None:
                return name, prefix
            out = fn(path)
            return name, prefix + out
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
