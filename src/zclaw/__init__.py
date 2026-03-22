"""ZCLAW: local AI assistant demo (vLLM + workspace context)."""

from zclaw.llm import VLLMChatModel
from zclaw.prompts import SECOND_STAGE_PROMPT, TOOL_CALL_PROMPT_OPTIMIZED
from zclaw.tool_loop import (
    ToolLoopResult,
    build_tool_call_prompt,
    execute_tool_call,
    parse_tool_call,
    resolve_path,
    resolve_path_with_meta,
    run_tool_loop,
)
from zclaw.tools import get_file_content, get_project_directory
from zclaw.workspace import (
    build_workspace_digest,
    build_system_prompt_with_workspace,
    resolve_workspace_root,
)

__all__ = [
    "VLLMChatModel",
    "TOOL_CALL_PROMPT_OPTIMIZED",
    "SECOND_STAGE_PROMPT",
    "ToolLoopResult",
    "build_tool_call_prompt",
    "build_workspace_digest",
    "build_system_prompt_with_workspace",
    "execute_tool_call",
    "get_file_content",
    "get_project_directory",
    "parse_tool_call",
    "resolve_path",
    "resolve_path_with_meta",
    "resolve_workspace_root",
    "run_tool_loop",
]
