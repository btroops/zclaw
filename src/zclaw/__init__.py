"""ZCLAW: local AI assistant demo (vLLM + workspace context)."""

from zclaw.llm import VLLMChatModel
from zclaw.workspace import (
    build_workspace_digest,
    build_system_prompt_with_workspace,
    resolve_workspace_root,
)

__all__ = [
    "VLLMChatModel",
    "build_workspace_digest",
    "build_system_prompt_with_workspace",
    "resolve_workspace_root",
]
