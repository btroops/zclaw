"""Tests for natural-language path intent correction."""

from __future__ import annotations

from zclaw.path_intent import (
    apply_mutation_path_intent,
    extract_in_file_path_from_instruction,
    extract_under_create_directory,
    extract_under_create_file,
    extract_under_delete_directory,
    extract_under_write_file,
    maybe_align_file_path_with_user_instruction,
    maybe_coerce_create_directory_to_create_file_from_user_intent,
    maybe_coerce_directory_to_create_file,
    path_last_segment_looks_like_file,
)


def test_extract_under_delete_directory_src_functest() -> None:
    r = extract_under_delete_directory("在src目录删除functest目录")
    assert r is not None
    assert r[0] == "src"
    assert r[1] == "functest"


def test_apply_delete_directory_corrects_leaf_only() -> None:
    p, note = apply_mutation_path_intent(
        "在src目录删除functest目录",
        "delete_directory",
        {"dir_path": "functest"},
    )
    assert p["dir_path"] == "src/functest"
    assert "路径校正" in note


def test_extract_under_create_file_loose_java() -> None:
    r = extract_under_create_file("在functest目录下创建Hello.java")
    assert r is not None
    assert r[0] == "functest"
    assert r[1] == "Hello.java"


def test_extract_under_create_file_inline_no_xia_or_dir() -> None:
    """「在 src/functest 创建 HelloWorld.java」无「下」「目录」。"""
    r = extract_under_create_file("在src/functest创建HelloWorld.java")
    assert r is not None
    assert r[0] == "src/functest"
    assert r[1] == "HelloWorld.java"


def test_extract_in_file_path_from_instruction() -> None:
    assert (
        extract_in_file_path_from_instruction("在src/functest/HelloWold.java中写快速排序")
        == "src/functest/HelloWold.java"
    )


def test_maybe_align_file_path_fixes_typo() -> None:
    p, note = maybe_align_file_path_with_user_instruction(
        "在src/functest/HelloWold.java中写",
        "write_file",
        {"file_path": "src/functest/HelloWorld.java", "content": "x"},
    )
    assert p["file_path"] == "src/functest/HelloWold.java"
    assert "对齐" in note


def test_extract_under_create_file_dir_without_lixia() -> None:
    """「在 src/functest 目录创建一个 …」无「里/下」。"""
    r = extract_under_create_file("在src/functest目录创建一个HelloWold.java")
    assert r is not None
    assert r[0] == "src/functest"
    assert r[1] == "HelloWold.java"


def test_extract_under_create_directory_rejects_java_leaf() -> None:
    assert extract_under_create_directory("在functest目录下创建Hello.java") is None


def test_maybe_coerce_create_directory_from_user_intent_inline_create() -> None:
    n, p, note = maybe_coerce_create_directory_to_create_file_from_user_intent(
        "在src/functest创建HelloWorld.java",
        "create_directory",
        {"dir_path": "src/functest"},
    )
    assert n == "create_file"
    assert p["file_path"] == "src/functest/HelloWorld.java"
    assert "工具校正" in note


def test_maybe_coerce_create_directory_from_user_intent_to_create_file() -> None:
    n, p, note = maybe_coerce_create_directory_to_create_file_from_user_intent(
        "在src/functest目录创建一个HelloWold.java",
        "create_directory",
        {"dir_path": "src/functest"},
    )
    assert n == "create_file"
    assert p["file_path"] == "src/functest/HelloWold.java"
    assert "工具校正" in note


def test_maybe_coerce_create_directory_to_create_file() -> None:
    n, p, note = maybe_coerce_directory_to_create_file(
        "create_directory",
        {"dir_path": "functest/Hello.java"},
    )
    assert n == "create_file"
    assert p["file_path"] == "functest/Hello.java"
    assert "工具校正" in note


def test_path_last_segment_looks_like_file() -> None:
    assert path_last_segment_looks_like_file("Hello.java") is True
    assert path_last_segment_looks_like_file("foo") is False


def test_extract_under_create_directory_src_functest() -> None:
    t = "在src目录下创建一个functest目录"
    r = extract_under_create_directory(t)
    assert r is not None
    assert r[0] == "src"
    assert r[1] == "functest"


def test_extract_under_create_directory_spaced() -> None:
    r = extract_under_create_directory("在 src/zclaw 下创建一个 testdir 目录")
    assert r is not None
    assert r[0] == "src/zclaw"
    assert r[1] == "testdir"


def test_apply_create_directory_corrects_leaf_only() -> None:
    p, note = apply_mutation_path_intent(
        "在src目录下创建一个functest目录",
        "create_directory",
        {"dir_path": "functest"},
    )
    assert p["dir_path"] == "src/functest"
    assert "路径校正" in note


def test_apply_create_directory_noop_when_already_full() -> None:
    p, note = apply_mutation_path_intent(
        "在src目录下创建一个functest目录",
        "create_directory",
        {"dir_path": "src/functest"},
    )
    assert p["dir_path"] == "src/functest"
    assert note == ""


def test_extract_under_write_file() -> None:
    r = extract_under_write_file("在src目录下写入notes.txt")
    assert r is not None
    assert r == ("src", "notes.txt")


def test_extract_under_create_file() -> None:
    r = extract_under_create_file("在pkg目录下创建一个hello.py文件")
    assert r is not None
    assert r[0] == "pkg"
    assert r[1] == "hello.py"


def test_apply_write_file_from_instruction() -> None:
    p, note = apply_mutation_path_intent(
        "在src目录下写入log.txt",
        "write_file",
        {"file_path": "log.txt", "content": "x"},
    )
    assert p["file_path"] == "src/log.txt"
    assert "路径校正" in note
