"""Two-stage flow: model emits tool JSON → execute tools → model synthesizes reply."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from zclaw.prompts import SECOND_STAGE_PROMPT, TOOL_CALL_PROMPT_OPTIMIZED
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
    return TOOL_CALL_PROMPT_OPTIMIZED.format(
        default_root_dir=default_root_dir,
        user_instruction=user_instruction,
    )


def extract_json_object(text: str) -> str:
    """Strip fences and isolate the first JSON object in ``text``."""
    s = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def parse_tool_call(text: str) -> Dict[str, Any]:
    raw = extract_json_object(text)
    return json.loads(raw)


def resolve_path(user_path: str, default_root: str) -> str:
    p = user_path.strip()
    if not p:
        return default_root
    from pathlib import Path

    exp = Path(p).expanduser()
    if exp.is_absolute():
        return str(exp.resolve())
    return str((Path(default_root) / exp).resolve())


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
            root_dir = params.get("root_dir") or default_root_dir
            root_dir = resolve_path(str(root_dir), default_root_dir)
            exclude = params.get("exclude_dirs")
            md = int(params.get("max_depth", 5))
            return name, fn(
                root_dir,
                exclude_dirs=exclude,
                max_depth=md,
            )
        if name == "get_file_content":
            fp = params.get("file_path")
            if fp is None or not str(fp).strip():
                return name, "错误：未提供 file_path（文件绝对路径或相对默认工程根目录的路径）。"
            fp = resolve_path(str(fp).strip(), default_root_dir)
            return name, fn(fp)
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
