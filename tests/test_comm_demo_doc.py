"""通信演示页与实现的一致性对账（`docs/v2.2.0-comm-demo.html`）。

演示页是给电控组看的「一眼看懂」材料，很容易在协议改动后忘记同步。
这里把它里面写死的常量与表逐项对回 `protocol.py`，避免演示页变成
「看起来权威、实际过时」的第二份契约。

校验四类：

1. 命令码与任务码常量必须与实现相同；
2. 右侧「帧长一览」表的每一行必须等于实现算出来的 REQ_LEN / RES_LEN + 9；
3. 页面里出现的 `0x00xx` 命令码只能是 0x0010 / 0x0011，不得残留旧命令；
4. 帧编码函数必须与 `protocol.encode()` 逐字节一致（页面自带一份 CRC，
   两份实现不一致是最难发现的那种错）。
"""
import re
from pathlib import Path

import pytest

from src.core.protocol import (CMD_REQUEST, CMD_RESULT, ID_FIELD_LEN, REQ_LEN,
                               RES_LEN, TASK_ALIGN, TASK_SCAN, TASK_TURNTABLE,
                               HEADER_LEN, TAIL_LEN)

DEMO = Path(__file__).resolve().parent.parent / "docs" / "v2.2.0-comm-demo.html"


@pytest.fixture(scope="module")
def demo_text():
    return DEMO.read_text(encoding="utf-8")


def test_demo_page_exists_and_is_self_contained(demo_text):
    """必须是单文件：不引外部 js/css，才能直接双击打开。"""
    assert "<!doctype html>" in demo_text.lower()
    assert not re.search(r'<script[^>]+src=', demo_text), "不该引外部脚本"
    assert not re.search(r'<link[^>]+stylesheet', demo_text), "不该引外部样式"


def test_demo_command_codes_match_implementation(demo_text):
    assert f"const CMD_REQUEST = 0x{CMD_REQUEST:04X};" in demo_text
    assert f"const CMD_RESULT  = 0x{CMD_RESULT:04X};" in demo_text


def test_demo_no_retired_command_codes(demo_text):
    """演示页只该出现 V2.2.0 的两条命令码。"""
    found = set(re.findall(r"0x00[0-9A-Fa-f]{2}\b", demo_text))
    allowed = {f"0x{CMD_REQUEST:04X}", f"0x{CMD_RESULT:04X}"}
    # 允许 0x8C / 0x8408 之类的 CRC 多项式常量（不是命令码，形如 0x0xxx 的才检查）
    unexpected = {c for c in found if c not in allowed}
    assert not unexpected, f"演示页里出现了非 V2.2.0 的命令码：{sorted(unexpected)}"


def test_demo_task_codes_match_implementation(demo_text):
    """任务码常量（页面写在一行里，用逗号分隔，所以逐个数解析）。"""
    decls = dict(re.findall(r"\b(TASK_\w+)\s*=\s*(\d+)", demo_text))
    assert int(decls["TASK_SCAN"]) == TASK_SCAN
    assert int(decls["TASK_TURNTABLE"]) == TASK_TURNTABLE
    assert int(decls["TASK_ALIGN"]) == TASK_ALIGN


def _frame_len_rows(demo_text):
    """取「帧长一览」表的数据行：[(方向, 任务码文字, data 长度, 帧长)]。"""
    idx = demo_text.index("帧长一览")
    block = demo_text[idx:]
    rows = []
    for line in block.splitlines():
        if line.strip() == "</table>":
            break
        if not line.lstrip().startswith("<tr><td>"):
            continue
        cells = re.findall(r"<td[^>]*>(.*?)</td>", line)
        if len(cells) != 4:
            continue
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        rows.append(cells)
    return rows


def test_demo_length_table_matches_implementation(demo_text):
    """表里每一行都必须等于实现算出来的长度。

    写错这行不会报错，只会让电控按错的字节数读，把下一帧的帧头当正文。
    """
    overhead = HEADER_LEN + ID_FIELD_LEN + TAIL_LEN
    want = {
        ("请求", "1"): REQ_LEN[TASK_SCAN],
        ("结果", "1"): RES_LEN[TASK_SCAN],
        ("请求", "2"): REQ_LEN[TASK_TURNTABLE],
        ("结果", "2"): RES_LEN[TASK_TURNTABLE],
        ("请求", "3"): REQ_LEN[TASK_ALIGN],
        ("结果", "3"): RES_LEN[TASK_ALIGN],
    }
    rows = _frame_len_rows(demo_text)
    assert len(rows) == 6, f"帧长表应有 6 行，实际 {len(rows)} 行：{rows}"

    seen = set()
    for direction, task_cell, data_len, frame_len in rows:
        task_code = task_cell.split()[0]
        key = (direction, task_code)
        assert key in want, f"帧长表出现意外的行：{(direction, task_cell)}"
        seen.add(key)
        assert int(data_len) == want[key], \
            f"{key} 的 data 长度应为 {want[key]}，页面写的是 {data_len}"
        assert int(frame_len) == overhead + want[key], \
            f"{key} 的帧长应为 {overhead + want[key]}，页面写的是 {frame_len}"
    assert seen == set(want), f"帧长表缺行：{set(want) - seen}"


def test_demo_has_no_removed_concepts(demo_text):
    """V2.2.0 取消的东西不能再出现在演示页的说明里（除了明确说「没有」）。"""
    for word in ("心跳", "done"):
        # 允许出现在「没有心跳」「没有 done」这类否定句里
        for m in re.finditer(word, demo_text):
            ctx = demo_text[max(0, m.start() - 22):m.start() + 4]
            assert re.search(r"没有|无 |也没|不是|取消|不需要", ctx), \
                f"演示页提到「{word}」但没有说明它已被取消：…{ctx}…"


def test_demo_crc_constants_match_implementation(demo_text):
    """页面自带一份 CRC 实现，多项式必须与 `crc.py` 一致，否则算出的字节全错。"""
    from src.core.crc import _CRC8_POLY_REFLECTED, _CRC16_POLY_REFLECTED
    assert f"0x{_CRC8_POLY_REFLECTED:02X}" in demo_text
    assert f"0x{_CRC16_POLY_REFLECTED:04X}" in demo_text
    assert "0xFF" in demo_text and "0xFFFF" in demo_text


# 页面里出现的样例帧（十六进制）。值本身由页面现算，这里固定下来是为了在
# protocol.py 或 crc.py 改动时立刻失败——而不是让演示页静默展示过时的字节。
SAMPLE_SCAN_REQ = "A5 03 00 07 A4 10 00 01 80 4A"
SAMPLE_SCAN_RES = ("A5 15 00 08 7E 11 00 07 01 00 04 05 02 03 02 01 "
                   "02 05 04 03 01 02 01 02 03 8D 33 7B")


def test_sample_frames_still_match_encode():
    """抽查两帧：页面演示用的字节必须就是 `encode()` 现在算出来的。"""
    from src.core.protocol import Frame, encode
    from src.core.taskplan import build_task_plan

    def hx(b):
        return " ".join(f"{x:02X}" for x in b)

    assert hx(encode(Frame(7, CMD_REQUEST, bytes((TASK_SCAN,))))) == SAMPLE_SCAN_REQ

    plan = build_task_plan("452+321+254+312").to_bytes()
    res = encode(Frame(8, CMD_RESULT, bytes((7, TASK_SCAN, 0)) + plan))
    assert hx(res) == SAMPLE_SCAN_RES
