"""CLI: print workspace digest, or run a short chat with workspace-aware system prompt."""

from __future__ import annotations

import argparse
import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from zclaw.llm import VLLMChatModel
from zclaw.workspace import build_system_prompt_with_workspace, build_workspace_digest


def _cmd_digest(args: argparse.Namespace) -> int:
    text = build_workspace_digest(
        args.root,
        max_tree_depth=args.tree_depth,
        max_tree_lines=args.tree_lines,
        readme_max_chars=args.readme_chars,
    )
    print(text)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    base_url = args.base_url or os.environ.get("ZCLAW_VLLM_BASE", "http://127.0.0.1:8000/v1")
    model_name = args.model or os.environ.get("ZCLAW_MODEL", "Qwen/Qwen2.5-7B")

    system_text = build_system_prompt_with_workspace(
        args.system,
        root=args.root,
        max_tree_depth=args.tree_depth,
        max_tree_lines=args.tree_lines,
        readme_max_chars=args.readme_chars,
    )
    llm = VLLMChatModel(
        base_url=base_url,
        model_name=model_name,
        api_key=args.api_key or os.environ.get("ZCLAW_VLLM_API_KEY", "token-abc123"),
        max_tokens=args.max_tokens,
    )

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=args.message),
    ]
    print("----- assistant (stream) -----\n")
    for chunk in llm.stream(messages):
        c = getattr(chunk, "content", None)
        if c is None and hasattr(chunk, "message"):
            c = getattr(chunk.message, "content", None)
        if c:
            print(c, end="", flush=True)
    print("\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="zclaw", description="ZCLAW workspace + vLLM demo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    root_kw = dict(
        default=None,
        help="Workspace root (default: $ZCLAW_WORKSPACE or cwd)",
    )

    p_digest = sub.add_parser("digest", help="Print workspace digest to stdout")
    p_digest.add_argument("--root", **root_kw)
    p_digest.add_argument("--tree-depth", type=int, default=2)
    p_digest.add_argument("--tree-lines", type=int, default=120)
    p_digest.add_argument("--readme-chars", type=int, default=4000)
    p_digest.set_defaults(func=_cmd_digest)

    p_chat = sub.add_parser("chat", help="One-shot chat with workspace context (needs vLLM)")
    p_chat.add_argument("--root", **root_kw)
    p_chat.add_argument("message", help="User message")
    p_chat.add_argument(
        "--system",
        default="你是一个有帮助的助手，用中文回答。",
        help="Base system instruction (workspace info is appended)",
    )
    p_chat.add_argument("--base-url", default=None, help="vLLM OpenAPI base, e.g. http://host:8000/v1")
    p_chat.add_argument("--model", default=None, help="Model id on the server")
    p_chat.add_argument("--api-key", default=None, help="Bearer token (or $ZCLAW_VLLM_API_KEY)")
    p_chat.add_argument("--max-tokens", type=int, default=1000)
    p_chat.add_argument("--tree-depth", type=int, default=2)
    p_chat.add_argument("--tree-lines", type=int, default=120)
    p_chat.add_argument("--readme-chars", type=int, default=4000)
    p_chat.set_defaults(func=_cmd_chat)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
