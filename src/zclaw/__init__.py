"""ZCLAW: local AI assistant demo (vLLM + workspace context)."""

from zclaw.llm import VLLMChatModel
from zclaw.path_resolve import (
    resolve_target_directory,
    resolve_target_file,
    resolve_under_workspace_write_chain,
    resolve_write_target_directory,
    resolve_write_target_file,
)
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
from zclaw.tools import (
    append_file,
    create_directory,
    create_file,
    delete_directory,
    delete_file,
    get_file_content,
    get_project_directory,
    rename_file,
    write_file,
)
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
    "append_file",
    "create_directory",
    "create_file",
    "delete_directory",
    "delete_file",
    "execute_tool_call",
    "get_file_content",
    "get_project_directory",
    "rename_file",
    "resolve_under_workspace_write_chain",
    "resolve_write_target_directory",
    "resolve_write_target_file",
    "write_file",
    "parse_tool_call",
    "resolve_path",
    "resolve_path_with_meta",
    "resolve_target_directory",
    "resolve_target_file",
    "resolve_workspace_root",
    "run_tool_loop",
]
