"""校验 docs/protocol-v0.2.md 里的字节级示例与实现一致（文档不可说谎）。

从文档里按**出现顺序**抽出所有 hex 代码块，用 ``decode_frame`` 解回 Frame，
再与下面的期望表逐条比对。这样任何人改协议实现而忘了同步文档，测试就红；
反过来文档被手改错一个字节也会被抓到。
"""
import re
from pathlib import Path

import pytest

from src.core.protocol import (CMD_ACK, CMD_CALIBRATE, CMD_DONE, CMD_GOTO,
                               CMD_HEARTBEAT, CMD_PICK, CMD_PLACE, CMD_POSE,
                               CMD_STATE, CMD_TASK_PLAN, Frame, decode_frame)

DOC = Path(__file__).resolve().parent.parent / "docs" / "protocol-v0.2.md"


def i16(v):
    return int(v).to_bytes(2, "little", signed=True)


# 文档 §4 中 hex 代码块的出现顺序，与各小节标题一一对应（顺序即校验）。
EXPECTED = [
    ("TASK_PLAN", Frame(1, CMD_TASK_PLAN,
                        bytes((4, 5, 2, 3, 2, 1, 2, 5, 4, 3, 1, 2)))),
    ("GOTO", Frame(2, CMD_GOTO, i16(1200) + i16(300) + i16(450))),
    ("GOTO 实例(NAV)", Frame(8, CMD_GOTO, i16(1200) + i16(900) + i16(0))),
    ("CALIBRATE", Frame(3, CMD_CALIBRATE, i16(-12) + i16(45) + i16(83))),
    ("PLACE", Frame(4, CMD_PLACE, b"\x01" + i16(-15) + i16(30))),
    ("PICK", Frame(5, CMD_PICK, b"\x02" + i16(10) + i16(-20))),
    ("HEARTBEAT", Frame(6, CMD_HEARTBEAT, b"")),
    ("ACK", Frame(1, CMD_ACK, b"\x02")),
    ("POSE", Frame(0, CMD_POSE,
                   i16(1520) + i16(300) + i16(-1234) + (1).to_bytes(2, "little"))),
    ("DONE", Frame(0, CMD_DONE, b"\x07")),
    ("STATE", Frame(0, CMD_STATE, b"\x03")),
    ("GOTO 边界值", Frame(7, CMD_GOTO, i16(32767) + i16(-32768) + i16(-18000))),
]


def _doc_text():
    if not DOC.is_file():
        pytest.skip(f"缺少协议文档 {DOC}")
    return DOC.read_text(encoding="utf-8")


def _hex_blocks(text):
    """文档里的 hex 代码块（单行、用 ``|`` 分组）→ 字节串列表，按出现顺序。"""
    blocks = []
    for raw in re.findall(r"```[a-z]*\n(.*?)```", text, re.DOTALL):
        chunky = raw.strip()
        if not chunky.lower().startswith("a5"):
            continue  # 跳过 python 命令、ASCII 帧结构图等
        tokens = chunky.replace("|", " ").split()
        try:
            blocks.append(bytes(int(tok, 16) for tok in tokens))
        except ValueError:
            continue
    return blocks


def test_doc_example_count_matches_expected():
    blocks = _hex_blocks(_doc_text())
    assert len(blocks) == len(EXPECTED), (
        f"文档有 {len(blocks)} 个 hex 示例，期望 {len(EXPECTED)} 个")


@pytest.mark.parametrize("label,frame", EXPECTED, ids=[e[0] for e in EXPECTED])
def test_doc_example_matches_implementation(label, frame):
    blocks = _hex_blocks(_doc_text())
    index = [e[0] for e in EXPECTED].index(label)
    raw = blocks[index]
    decoded = decode_frame(raw)
    assert decoded is not None, f"{label} 示例帧 CRC 不通过：{raw.hex(' ')}"
    assert decoded == frame, (
        f"{label} 文档与实现不一致：\n"
        f"  文档 {raw.hex(' ')}\n  期望 {frame}\n  实际 {decoded}")


def test_doc_examples_pass_crc_recompute():
    # 独立于 decode_frame：重新编码期望帧必须逐字节等于文档里的示例
    from src.core.protocol import encode

    blocks = _hex_blocks(_doc_text())
    for (label, frame), raw in zip(EXPECTED, blocks):
        assert encode(frame) == raw, f"{label} 文档示例与 encode() 输出不同"
