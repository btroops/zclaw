"""
判断 ``append_file`` 是否在**已有文件**末尾产生明显不合理的结构拼接。

依据「文件尾部是否已闭合」+「追加开头是否像新的独立文档/顶层作用域」做通用检查，
覆盖 JSON、HTML、花括号类源码等；不单独针对某一种语言写死逻辑。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_TAIL_MAX = 16384
_HEAD_MAX = 8192

# 花括号类源码常见后缀（闭合 `}` 后不应再贴顶层类型/入口）
_BRACE_SOURCE_SUFFIXES = frozenset(
    {
        ".java",
        ".cs",
        ".cpp",
        ".cc",
        ".cxx",
        ".c",
        ".h",
        ".hpp",
        ".hh",
        ".m",
        ".mm",
        ".go",
        ".rs",
        ".kt",
        ".kts",
        ".swift",
        ".scala",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".php",
        ".dart",
        ".vue",
        ".svelte",
    }
)

_JSON_SUFFIXES = frozenset({".json"})
_HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})

# 追加内容开头：像新的顶层类型 / 命名空间 / 包声明
_TOP_SCOPE_START = re.compile(
    r"^\s*(?:@[\w.]+\s+)*(?:export\s+)?(?:default\s+)?"
    r"(?:public\s+|private\s+|protected\s+|internal\s+)?"
    r"(?:abstract\s+|sealed\s+|open\s+|static\s+|final\s+|data\s+)*"
    r"(?:class|interface|enum|struct|record|namespace)\s+\w+",
    re.MULTILINE | re.IGNORECASE,
)
_PACKAGE_START = re.compile(r"^\s*package\s+[\w.]+", re.MULTILINE)

# 第二个程序入口（多语言常见形态）
_SECOND_ENTRYPOINT = re.compile(
    r"public\s+static\s+void\s+main\s*\(|"
    r"static\s+void\s+Main\s*\(|"
    r"\bfun\s+main\s*\(",
    re.IGNORECASE,
)


def _read_tail(path: Path, max_chars: int = _TAIL_MAX) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _generic_rewrite_hint() -> str:
    return (
        "请先 `get_file_content` 读取全文，在逻辑上合并后使用 `write_file` 一次写回完整内容；"
        "若只需在同一段落/同一结构内续写，请避免在追加开头出现新的独立文档或顶层声明。"
    )


def append_content_rejection_reason(file_path: str, content: str) -> Optional[str]:
    """
    若追加会明显破坏结构化文本的完整性，返回中文错误说明；否则 ``None``。

    目标文件不存在时（将新建）不做结构校验。
    """
    if content is None or not str(content).strip():
        return None

    path = Path(file_path)
    if not path.is_file():
        return None

    tail = _read_tail(path)
    head = str(content).lstrip()[:_HEAD_MAX]
    if not head:
        return None

    tail_st = tail.rstrip()
    head_st = head.lstrip()
    suf = path.suffix.lower()

    # ----- JSON：已结束的根值后再接一个根对象/数组 -----
    if suf in _JSON_SUFFIXES and tail_st:
        if tail_st.endswith("}") or tail_st.endswith("]"):
            if head_st.startswith("{") or head_st.startswith("["):
                return (
                    "错误：JSON 文件已到闭合的 `}` 或 `]`，再追加新的顶层 `{`/`[` 会得到非法的多根 JSON。"
                    + _generic_rewrite_hint()
                )

    # ----- HTML：已出现 </html> 后再接新文档 -----
    if suf in _HTML_SUFFIXES and tail_st:
        if re.search(r"</html>\s*$", tail, re.IGNORECASE):
            if re.match(r"^\s*<!DOCTYPE\b", head_st, re.IGNORECASE) or re.match(
                r"^\s*<html\b", head_st, re.IGNORECASE
            ):
                return (
                    "错误：HTML 在 `</html>` 之后不应再追加新的 `<!DOCTYPE` 或顶层 `<html>`。"
                    + _generic_rewrite_hint()
                )

    # ----- 花括号类源码：尾部已 `}` 闭合，追加却像新顶层单元或第二入口 -----
    if suf in _BRACE_SOURCE_SUFFIXES and tail_st.endswith("}"):
        if _TOP_SCOPE_START.search(head_st):
            return (
                "错误：文件末尾已出现顶层块闭合 `}`，追加内容却以新的类型/接口/命名空间等顶层声明开头，"
                "极易产生非法源码结构。" + _generic_rewrite_hint()
            )
        if _PACKAGE_START.match(head_st):
            return (
                "错误：文件已闭合后不应再追加 `package` 声明。" + _generic_rewrite_hint()
            )
        if _SECOND_ENTRYPOINT.search(head_st):
            return (
                "错误：追加内容包含第二个程序入口（如 `main`），应在原有入口或结构内扩展，"
                "或合并全文后 `write_file`。" + _generic_rewrite_hint()
            )

    return None
