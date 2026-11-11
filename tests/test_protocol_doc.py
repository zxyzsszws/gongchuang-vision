"""协议文档（`docs/protocol-v1.0.0.md`）与实现的一致性对账。

文档是交给电控组实现的契约，一旦与代码不符就是事故。这里做三类校验：

1. **帧格式表**：表 1-2 / 1-3 / 1-4 的字段名、偏移、大小必须与 `protocol.py` 常量一致。
2. **命令码表**：表 1-5 的每一行（命令码、数据段长度）必须与 `CMD_*` / `DATA_LEN` 一致。
3. **样例字节**：§1.4 的每个 hex 代码块必须能被 `decode_frame` 解回对应 Frame，
   且 `encode()` 重新编码后逐字节等于文档中的示例。
"""
import re
from pathlib import Path

import pytest

from src.core import protocol as P
from src.core.protocol import (CMD_ACK, CMD_CALIBRATE, CMD_DONE, CMD_GOTO,
                               CMD_HEARTBEAT, CMD_PICK, CMD_PLACE, CMD_POSE,
                               CMD_STATE, CMD_TASK_PLAN, Frame, decode_frame,
                               encode)

DOC = Path(__file__).resolve().parent.parent / "docs" / "protocol-v1.0.0.md"


def i16(v):
    return int(v).to_bytes(2, "little", signed=True)


def u16(v):
    return int(v).to_bytes(2, "little")


# §1.4 中 hex 代码块的出现顺序 → 期望的 Frame
EXPECTED_SAMPLES = [
    ("0x0002 TASK_PLAN", Frame(1, CMD_TASK_PLAN,
                               bytes((4, 5, 2, 3, 2, 1, 2, 5, 4, 3, 1, 2)))),
    ("0x0003 GOTO", Frame(2, CMD_GOTO, i16(1200) + i16(300) + i16(450))),
    ("0x0003 GOTO 导航实例", Frame(8, CMD_GOTO, i16(1200) + i16(900) + i16(0))),
    ("0x0004 CALIBRATE", Frame(3, CMD_CALIBRATE, i16(-12) + i16(45) + i16(83))),
    ("0x0005 PLACE", Frame(4, CMD_PLACE, b"\x01" + i16(-15) + i16(30))),
    ("0x0006 PICK", Frame(5, CMD_PICK, b"\x02" + i16(10) + i16(-20))),
    ("0x0001 HEARTBEAT", Frame(6, CMD_HEARTBEAT, b"")),
    ("0x0007 ACK", Frame(1, CMD_ACK, b"\x02")),
    ("0x0201 POSE", Frame(0, CMD_POSE, i16(1520) + i16(300) + i16(-1234) + u16(1))),
    ("0x0203 DONE", Frame(0, CMD_DONE, b"\x07")),
    ("0x0202 STATE", Frame(0, CMD_STATE, b"\x03")),
    ("0x0003 边界值", Frame(7, CMD_GOTO, i16(32767) + i16(-32768) + i16(-18000))),
]

# 表 1-5 中声明的（命令码, 数据段长度）
EXPECTED_CMD_TABLE = {
    0x0001: 0, 0x0002: 12, 0x0003: 6, 0x0004: 6, 0x0005: 5, 0x0006: 5,
    0x0007: 1, 0x0201: 8, 0x0202: 1, 0x0203: 1,
}


@pytest.fixture(scope="module")
def doc_text():
    if not DOC.is_file():
        pytest.skip(f"缺少协议文档 {DOC}")
    return DOC.read_text(encoding="utf-8")


def _hex_blocks(text):
    """文档里「a5 开头、用 | 分组」的 hex 代码块，按出现顺序返回字节串。"""
    blocks = []
    for raw in re.findall(r"```[a-z]*\n(.*?)```", text, re.DOTALL):
        body = raw.strip()
        if not body.lower().startswith("a5"):
            continue
        tokens = body.replace("|", " ").split()
        try:
            blocks.append(bytes(int(tok, 16) for tok in tokens))
        except ValueError:
            continue
    return blocks


# ---------- 1. 帧格式表 ----------

def test_doc_states_frame_layout_tables(doc_text):
    for heading in ("表 1-2 通信协议格式", "表 1-3 frame_header 格式",
                    "表 1-4 帧头详细定义", "表 1-5 命令码 ID 一览"):
        assert heading in doc_text, f"文档缺少「{heading}」"


def test_header_field_offsets_match_implementation(doc_text):
    # 表 1-4 的四行：(域, 偏移, 大小)
    for field, offset, size in (("SOF", 0, 1), ("data_length", 1, 2),
                                ("seq", 3, 1), ("CRC8", 4, 1)):
        row = re.search(rf"\|\s*{field}\s*\|\s*{offset}\s*\|\s*{size}\s*\|",
                        doc_text)
        assert row, f"表 1-4 缺少 {field} 的偏移 {offset} / 大小 {size}"
    assert P.HEADER_LEN == 5
    assert P.CMD_ID_LEN == 2
    assert P.TAIL_LEN == 2
    assert P.OVERHEAD == 9


def test_doc_matches_layout_dimensions(doc_text):
    # 表 1-2 的 5-byte / 2-byte / n-byte / 2-byte
    assert "| frame_header | cmd_id | data | frame_tail |" in doc_text
    assert f"5-byte | 2-byte | n-byte | 2-byte" in doc_text.replace("| ", "| ")
    # 表 1-3 的 1-byte / 2-byte / 1-byte / 1-byte
    assert "SOF | data_length | seq | CRC8" in doc_text


def test_sof_value_in_doc(doc_text):
    assert "固定值为 0xA5" in doc_text
    assert P.SOF == 0xA5


def test_data_length_range_in_doc(doc_text):
    assert f"{P.MIN_DATA_LENGTH}~{P.MAX_DATA_LENGTH}" in doc_text


# ---------- 2. 命令码表 ----------

def test_cmd_table_rows_match_constants(doc_text):
    """表 1-5 每行的「命令码 | 数据段长度」必须与实现一致（双向核对）。"""
    rows = re.findall(r"^\|\s*(0x[0-9A-Fa-f]{4})\s*\|\s*(\d+)\s*\|", doc_text,
                      re.MULTILINE)
    parsed = {int(cmd, 16): int(length) for cmd, length in rows}
    assert parsed == EXPECTED_CMD_TABLE, (
        f"文档表 1-5 与期望不符：\n文档 {parsed}\n期望 {EXPECTED_CMD_TABLE}")
    # 实现的 DATA_LEN 必须与之完全相同
    assert P.DATA_LEN == EXPECTED_CMD_TABLE


def test_every_command_has_a_detail_table(doc_text):
    """每个命令码都要有对应的「表 1-N 0xXXXX」小节与 typedef，不能漏写。"""
    for cmd in EXPECTED_CMD_TABLE:
        assert re.search(rf"表 1-\d+\s+0x{cmd:04X}\b", doc_text), (
            f"文档缺少命令 0x{cmd:04X} 的详细定义表")
    typedefs = re.findall(r"typedef __packed struct", doc_text)
    assert len(typedefs) == len(EXPECTED_CMD_TABLE), (
        "typedef 数量应与命令数一致（每个命令一段）")


def test_all_command_constants_documented(doc_text):
    """反向核对：代码里定义的 CMD_* 都必须在表 1-5 出现，避免加了命令没写文档。"""
    for name in dir(P):
        if not name.startswith("CMD_"):
            continue
        value = getattr(P, name)
        assert f"0x{value:04X}" in doc_text, f"{name}=0x{value:04X} 未写入文档"


def test_cmd_direction_segments_in_doc(doc_text):
    assert "`0x00xx` 为视觉单元发往机器人主控" in doc_text
    assert "`0x02xx` 为机器人主控发往视觉单元" in doc_text
    # 实现侧：CMD_ 前半段是上行、CMD_POSE 起是下行
    assert CMD_HEARTBEAT == 0x0001 and CMD_ACK == 0x0007
    assert CMD_POSE == 0x0201 and CMD_DONE == 0x0203


# ---------- 3. 样例字节 ----------

def test_sample_count(doc_text):
    assert len(_hex_blocks(doc_text)) == len(EXPECTED_SAMPLES)


@pytest.mark.parametrize("label,frame", EXPECTED_SAMPLES,
                         ids=[s[0] for s in EXPECTED_SAMPLES])
def test_sample_decodes_to_expected_frame(doc_text, label, frame):
    index = [s[0] for s in EXPECTED_SAMPLES].index(label)
    raw = _hex_blocks(doc_text)[index]
    decoded = decode_frame(raw)
    assert decoded is not None, f"{label} 样例 CRC 不通过：{raw.hex(' ')}"
    assert decoded == frame, (
        f"{label} 文档与实现不一致：\n"
        f"  文档 {raw.hex(' ')}\n  期望 {frame}\n  实际 {decoded}")


def test_samples_equal_reencode(doc_text):
    """样例必须逐字节等于 encode() 输出（防止手改文档里的某个 CRC 字节）。"""
    for (label, frame), raw in zip(EXPECTED_SAMPLES, _hex_blocks(doc_text)):
        assert encode(frame) == raw, f"{label} 文档样例与 encode() 输出不同"


def test_sample_data_length_annotation(doc_text):
    """文档写「data 12 字节，data_length = 14」，实现对 TASK_PLAN 必须成立。"""
    assert "`data_length` = 14 = `0x000E`" in doc_text
    frame = EXPECTED_SAMPLES[0][1]
    assert len(frame.data) == 12
    assert P.CMD_ID_LEN + len(frame.data) == 14
    assert encode(frame)[1:3] == b"\x0e\x00"     # 小端


# ---------- 4. 附录一 CRC 代码 ----------

def test_appendix_crc_table_agrees_with_implementation(doc_text):
    """文档附录一的 CRC8 表首行与实现生成的表一致（防止贴错表）。"""
    from src.core.crc import CRC8_TAB, CRC16_TAB

    appendix = doc_text[doc_text.index("附录一"):]
    assert "0x00, 0x5e, 0xbc, 0xe2, 0x61, 0x3f, 0xdd, 0x83" in appendix
    assert "0x35," in appendix
    assert CRC8_TAB[0] == 0x00 and CRC8_TAB[1] == 0x5E and CRC8_TAB[255] == 0x35
    # CRC16 表首行
    assert "0x0000, 0x1189, 0x2312, 0x329b" in appendix
    assert CRC16_TAB[0] == 0x0000 and CRC16_TAB[1] == 0x1189


def test_appendix_documents_both_init_values(doc_text):
    from src.core.crc import CRC16_INIT, CRC8_INIT
    assert "CRC8_INIT = 0xff" in doc_text and CRC8_INIT == 0xFF
    assert "CRC_INIT = 0xffff" in doc_text and CRC16_INIT == 0xFFFF
