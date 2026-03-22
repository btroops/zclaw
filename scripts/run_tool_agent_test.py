#!/usr/bin/env python3
"""
可选的端到端脚本：连接真实 vLLM，跑「工具 JSON → 本地工具 → 整合回复」闭环。

依赖：已 ``pip install -e .``，且能访问 OpenAI 兼容接口。

用法示例::

    export ZCLAW_VLLM_BASE=http://127.0.0.1:8000/v1
    export ZCLAW_WORKSPACE=/path/to/zclaw_demo
    python scripts/run_tool_agent_test.py "查看工程根目录下有哪些文件"

无服务时请用 pytest： ``pytest tests/ -q``
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    argv = [a for a in sys.argv[1:] if a]
    verbose = "-v" in argv or "--verbose" in argv
    argv = [a for a in argv if a not in ("-v", "--verbose")]

    if not argv:
        print(__doc__, file=sys.stderr)
        print(
            '示例: python scripts/run_tool_agent_test.py "读取 README.md 内容"',
            file=sys.stderr,
        )
        print("  -v / --verbose  将阶段一 JSON 与工具输出打到 stderr", file=sys.stderr)
        return 2

    root = os.path.abspath(os.path.expanduser(os.environ.get("ZCLAW_WORKSPACE", os.getcwd())))
    base = os.environ.get("ZCLAW_VLLM_BASE", "http://10.16.86.206:8000/v1")

    from zclaw.llm import VLLMChatModel
    from zclaw.tool_loop import run_tool_loop

    llm = VLLMChatModel(
        base_url=base,
        model_name=os.environ.get("ZCLAW_MODEL", "Qwen/Qwen2.5-7B"),
        api_key=os.environ.get("ZCLAW_VLLM_API_KEY", "token-abc123"),
    )
    msg = argv[0]
    result = run_tool_loop(llm, msg, default_root_dir=root)

    if verbose:
        print("===== stage1: raw tool JSON =====\n", file=sys.stderr)
        print(result.tool_call_raw, file=sys.stderr)
        print("\n===== tool =====\n", file=sys.stderr)
        print(f"name={result.tool_name!r}\n", file=sys.stderr)
        print(result.tool_output[:12000], file=sys.stderr)
        if len(result.tool_output) > 12000:
            print("\n… (truncated in script output)", file=sys.stderr)
        print("\n===== stage2: final =====\n", file=sys.stderr)
    print(result.final_reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
