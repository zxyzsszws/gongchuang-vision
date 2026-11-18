"""协议文档（`docs/protocol-v2.1.0.md`）与实现的一致性对账。

文档是交给电控组实现的契约，一旦与代码不符就是事故。这里做五类校验：

1. **帧格式表**：表 1-2 / 1-3 / 1-4 的字段名、偏移、大小必须与 `protocol.py` 常量一致。
2. **命令码表**：表 1-5 的每一行（命令码、数据段长度）必须与命令集一致，
   且每个命令都必须有「字节偏移量/大小/说明」详情表与一段 `typedef __packed struct`。
3. **样例字节**：§1.4 的每个 hex 代码块必须能被 `decode_frame` 解回对应 Frame，
   且 `encode()` 重新编码后逐字节等于文档中的示例。
4. **任务计划语义**：二维码四组数字的分组规则、以及第二批暂存区靶号按颜色配对
   的算法必须与文档示例逐项吻合（这段最容易在实现时写错且不报错）。
5. **V2.1.0 的新规则**：没有心跳、只有两类帧需要应答、「业务帧即应答」、
   重发时 seq 不变、解算必须按 echo_seq 去重。这几条写漏了，电控会照着实现出
   一个「每帧都回 ack」或「重复执行动作」的版本。
"""
import re
from pathlib import Path

import pytest

from src.core import protocol as P
from src.core.protocol import Frame, decode_frame, encode

DOC = Path(__file__).resolve().parent.parent / "docs" / "protocol-v2.1.0.md"


def i16(v):
    return int(v).to_bytes(2, "little", signed=True)


def u16(v):
    return int(v).to_bytes(2, "little")


# ---------- V2.1.0 命令集（表 1-5，不含「不使用」表） ----------

CMD_TASK_PLAN = 0x0002
CMD_POSE = 0x0003
CMD_TURNTABLE = 0x0004
CMD_REGION = 0x0005
CMD_ARM_STATE = 0x0006
CMD_ACK = 0x0007
CMD_ARM_SOLUTION = 0x0008

EXPECTED_CMD_TABLE = {
    CMD_TASK_PLAN: 16,
    CMD_POSE: 9,
    CMD_TURNTABLE: 16,
    CMD_REGION: 1,
    CMD_ARM_STATE: 7,
    CMD_ACK: 2,
    CMD_ARM_SOLUTION: 10,
}

# 已废弃、接收方须静默丢弃的命令（文档单列一张表）
RETIRED_CMDS = {0x0001, 0x0201, 0x0202, 0x0203}


# ---------- §1.4 样例：二维码 452+321+254+312 ----------
#
# 第一批: 颜色 4,5,2 -> 靶 3,2,1
# 第二批: 颜色 2,5,4 -> 粗加工区靶 3,1,2
# 第二批暂存区按颜色配对: 颜色2 在第一批的 1 号靶, 颜色5 在 2 号, 颜色4 在 3 号 -> 1,2,3
# 状态 0x8D = bit0(颜色能配上) + bit2(第2组是排列) + bit3(第4组是排列) + bit7(可用)
TASK_PLAN_DATA = bytes((4, 5, 2, 3, 2, 1, 2, 5, 4, 3, 1, 2, 1, 2, 3, 0x8D))

# 位姿: (1200, 600), 朝向 2.3deg, 残差 2.0mm, 质量 0xE3, 区域码 1(粗加工区)
POSE_DATA = i16(1200) + i16(600) + i16(23) + bytes((4, 0xE3, 1))

# 转盘: 圆心(250,-60), 半径150mm, 0号工位 60.0deg, 角速度 +60.0deg/s,
#       3 工位, 状态 0x8F, 三工位颜色 4,5,2, 预留
TURNTABLE_DATA = (i16(250) + i16(-60) + u16(150) + i16(600) + i16(600)
                  + bytes((3, 0x8F)) + bytes((4, 5, 2)) + bytes((0,)))

# 臂状态: l=600mm, h=300mm, yaw=15.0deg, 目标编号 1
ARM_STATE_DATA = u16(600) + u16(300) + i16(150) + bytes((1,))

# 应答: 确认 seq=11，结果 0(成功)
ACK_DATA = bytes((11, 0))

# 解算: d=120mm, alpha=30.0deg, 状态 0xA0(成功|质量好|可用), echo_seq=11,
#       目标点 (700, 100)mm
ARM_SOLUTION_DATA = (i16(120) + i16(300) + bytes((0xA0, 11))
                     + i16(700) + i16(100))

EXPECTED_SAMPLES = [
    ("TASK_PLAN", Frame(2, CMD_TASK_PLAN, TASK_PLAN_DATA)),
    ("POSE", Frame(3, CMD_POSE, POSE_DATA)),
    ("TURNTABLE", Frame(4, CMD_TURNTABLE, TURNTABLE_DATA)),
    ("REGION", Frame(5, CMD_REGION, bytes((2,)))),
    ("ARM_STATE", Frame(6, CMD_ARM_STATE, ARM_STATE_DATA)),
    ("ACK", Frame(6, CMD_ACK, ACK_DATA)),
    ("ARM_SOLUTION", Frame(7, CMD_ARM_SOLUTION, ARM_SOLUTION_DATA)),
]


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
    for heading in ("表 1-1 波特率", "表 1-2 通信协议格式",
                    "表 1-3 frame_header 格式", "表 1-4 帧头详细定义",
                    "表 1-5 命令码 ID 一览"):
        assert heading in doc_text, f"文档缺少「{heading}」"


def test_header_field_offsets_match_implementation(doc_text):
    for field, offset, size in (("SOF", 0, 1), ("data_length", 1, 2),
                                ("seq", 3, 1), ("CRC8", 4, 1)):
        row = re.search(rf"\|\s*{field}\s*\|\s*{offset}\s*\|\s*{size}\s*\|",
                        doc_text)
        assert row, f"表 1-4 缺少 {field} 的偏移 {offset} / 大小 {size}"
    assert P.HEADER_LEN == 5
    assert P.CMD_ID_LEN == 2
    assert P.TAIL_LEN == 2
    assert P.OVERHEAD == 9
    assert P.SOF == 0xA5


def test_doc_layout_dimensions(doc_text):
    """表 1-2 / 1-3 的字段与尺寸必须与实现一致。

    文档表格是 Markdown 对齐格式，单元格内有多余空格，故用**空白宽松**的正则，
    而不是逐字字符串包含——否则文档一重新排版测试就红，但它其实什么也没测到。
    """
    def row(*cells):
        return re.search(r"\|\s*" + r"\s*\|\s*".join(map(re.escape, cells))
                         + r"\s*\|", doc_text)

    assert row("frame_header", "cmd_id", "data", "frame_tail"), \
        "表 1-2 缺少 frame_header / cmd_id / data / frame_tail 一行"
    assert row("5-byte", "2-byte", "n-byte"), "表 1-2 的尺寸行缺失"
    assert row("SOF", "data_length", "seq", "CRC8"), "表 1-3 头部字段行缺失"
    assert row("1-byte", "2-byte", "1-byte"), "表 1-3 的尺寸行缺失"
    assert f"{P.MIN_DATA_LENGTH}~{P.MAX_DATA_LENGTH}" in doc_text


def test_broadcast_baudrate(doc_text):
    assert re.search(r"\|\s*视觉单元←→机器人主控\s*\|\s*115200\s*\|", doc_text)


# ---------- 2. 命令码表 ----------

def test_cmd_table_rows_match_contract(doc_text):
    """表 1-5 的「命令码 | 数据段长度」必须与约定完全一致（双向核对）。"""
    rows = re.findall(r"^\|\s*(0x[0-9A-Fa-f]{4})\s*\|\s*(\d+)\s*\|", doc_text,
                      re.MULTILINE)
    parsed = {int(cmd, 16): int(length) for cmd, length in rows}
    # 剔除「不使用」表里的行
    parsed = {k: v for k, v in parsed.items() if k not in RETIRED_CMDS}
    assert parsed == EXPECTED_CMD_TABLE, (
        f"文档表 1-5 与期望不符：\n文档 {parsed}\n期望 {EXPECTED_CMD_TABLE}")


def test_retired_commands_are_declared_unused(doc_text):
    """废弃命令必须被明确标注为「不使用」，否则电控会照着实现。"""
    section = doc_text[doc_text.index("本版本**不使用**"):]
    section = section[:section.index("## 1.3")]
    for cmd in RETIRED_CMDS:
        assert f"0x{cmd:04X}" in section, f"未声明 0x{cmd:04X} 不再使用"


def test_every_command_has_a_detail_table_and_typedef(doc_text):
    for cmd in EXPECTED_CMD_TABLE:
        assert re.search(rf"### 表 1-\d+ 0x{cmd:04X}\b", doc_text), (
            f"文档缺少命令 0x{cmd:04X} 的详细定义表")
    assert doc_text.count("typedef __packed struct") == len(EXPECTED_CMD_TABLE)


def test_every_command_has_its_data_length_documented(doc_text):
    """表 1-5 声明的长度必须能被该命令的详情表字段偏移+大小复现。"""
    for cmd, length in EXPECTED_CMD_TABLE.items():
        block_start = re.search(rf"### 表 1-\d+ 0x{cmd:04X}\b", doc_text).start()
        block_end = doc_text.find("### 表 1-", block_start + 10)
        if block_end < 0:
            block_end = doc_text.find("\n## ", block_start)
        block = doc_text[block_start:block_end]
        offsets = re.findall(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|", block, re.MULTILINE)
        assert offsets, f"0x{cmd:04X} 的详情表没有字段行"
        total = max(int(o) + int(s) for o, s in offsets)
        assert total == length, (
            f"0x{cmd:04X} 表 1-5 写 {length} 字节，字段表末位为 {total}")


def test_expected_table_matches_implementation_data_len():
    """本文件登记的期望表必须与 `protocol.DATA_LEN` 一致（防止两处各说各话）。

    文档 ↔ EXPECTED_CMD_TABLE 由 ``test_cmd_table_rows_match_contract`` 保证，
    这里补上 EXPECTED_CMD_TABLE ↔ 实现的另一半，构成完整的三方对账。
    """
    assert EXPECTED_CMD_TABLE == P.DATA_LEN
    assert RETIRED_CMDS <= P.RETIRED_CMDS


def test_region_hint_is_one_way_and_needs_no_ack(doc_text):
    """0x0005 是单向提示，不需要应答。"""
    assert "单向提示" in doc_text
    assert "仅在目标区域发生改变时发送一次" in doc_text
    assert "不需要应答" in doc_text


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
    """文档写「data 16 字节，data_length = 18 = 0x0012」，实现对 TASK_PLAN 必须成立。"""
    assert "`data_length` = 18 = `0x0012`" in doc_text
    assert len(TASK_PLAN_DATA) == 16
    assert P.CMD_ID_LEN + len(TASK_PLAN_DATA) == 18
    assert encode(EXPECTED_SAMPLES[0][1])[1:3] == b"\x12\x00"


# ---------- 4. 任务计划语义 ----------

def test_task_plan_sample_encodes_qr_452_321_254_312(doc_text):
    """样例里前 12 字节必须与二维码四组数字逐位对应。"""
    qr = "452+321+254+312"
    g1, g2, g3, g4 = qr.split("+")
    expect = bytes([int(c) for c in g1 + g2 + g3 + g4])
    assert TASK_PLAN_DATA[:12] == expect, (
        f"前 12 字节应为四组数字原样拼接 {expect.hex(' ')}")


def test_staging_ring_pairs_by_color(doc_text):
    """第二批暂存区靶号 = 在第一批中找同色物料的靶号；同批内颜色唯一故结果唯一。"""
    g1, g2, g3, _g4 = "452", "321", "254", "312"
    b1_color = [int(c) for c in g1]
    b1_ring = [int(c) for c in g2]
    b2_color = [int(c) for c in g3]

    # 同批内颜色互不相同 —— 这是配对唯一的充要条件，必须显式断言
    assert len(set(b1_color)) == len(b1_color)
    assert len(set(b2_color)) == len(b2_color)

    expect = [b1_ring[b1_color.index(c)] for c in b2_color]
    assert expect == [1, 2, 3]
    assert list(TASK_PLAN_DATA[12:15]) == expect, "文档样例的暂存区靶号与配对算法不符"


def test_task_plan_status_byte_bits(doc_text):
    """status 0x8D = 颜色能配上 | 第2组是排列 | 第4组是排列 | 可用。"""
    g2, g4 = "321", "312"
    assert sorted(int(c) for c in g2) == [1, 2, 3]
    assert sorted(int(c) for c in g4) == [1, 2, 3]
    status = TASK_PLAN_DATA[15]
    assert status & 0x80, "bit7 可用位必须置起"
    assert status & 0x01, "bit0 颜色能配上必须置起"
    assert status & 0x04, "bit2 第2组是排列必须置起"
    assert status & 0x08, "bit3 第4组是排列必须置起"
    assert status == 0x8D


def test_doc_documents_palletizing(doc_text):
    """码垛：第二批在暂存区必须叠在第一批之上且上下同色，文档不能漏。"""
    assert "码垛" in doc_text
    assert "上下两层颜色一致" in doc_text

    def row(*cells):
        return re.search(r"\|\s*" + r"\s*\|\s*".join(map(re.escape, cells))
                         + r"\s*\|", doc_text)

    # 示意图里第二批第 1 件颜色 2 -> 第一批同色在 1 号靶 -> 暂存区靶号 1
    assert row("第二批", "1", "2", "3", "1"), \
        "文档的码垛示意表缺少「第二批 1 号 颜色2 粗加工区3 暂存区1」这一行"


def test_doc_states_pose_replaces_not_adds(doc_text):
    """位姿是绝对量，主控须整体替换而非叠加——这条写错会导致长期偏差。"""
    assert "整体**替换**" in doc_text or "整体替换" in doc_text


def test_doc_states_single_target_cannot_solve_heading(doc_text):
    """必须讲清单靶缺朝向自由度，否则电控会以为一个靶就够。"""
    assert "旋转对称" in doc_text
    assert "缺少的正是朝向" in doc_text


# ---------- 5. V2.1.0 的新规则 ----------

def test_heartbeat_is_declared_removed(doc_text):
    """取消心跳必须在正文写明，并给出理由（否则会被当成漏写而加回来）。"""
    assert "本版本没有心跳" in doc_text
    assert "把静默当断链只能得到误报" in doc_text


def test_only_two_frames_require_ack(doc_text):
    """应答规则必须写清：周期测量帧与单向提示不应答。"""
    assert "只有这两类帧需要应答" in doc_text
    assert "不应答周期测量帧" in doc_text
    assert P.EXPECTS_ACK == {CMD_TASK_PLAN, CMD_ARM_SOLUTION}


def test_business_frame_as_ack_is_documented(doc_text):
    """0x0006 的应答是 0x0008，不另回 0x0007——这条不写清会做出两种等法。"""
    assert "业务帧即应答" in doc_text


def test_resend_keeps_seq_is_documented(doc_text):
    """重发时 seq 不变，且要与「seq 单调递增」讲清不冲突。"""
    assert "重发时 seq 不变" in doc_text
    assert "单调递增描述的是**新帧**" in doc_text


def test_duplicate_dedupe_by_echo_seq_is_mandatory(doc_text):
    """解算必须按 echo_seq 去重，否则车会沿同一个 d 走两遍。

    这类错误不产生任何报错，只表现为位置错，必须写成硬性要求。
    """
    assert "echo_seq" in doc_text
    assert "只执行一次" in doc_text
    assert "同一个 `shift_x` 走两遍" in doc_text


def test_ack_result_codes_are_documented(doc_text):
    """0x0007 的结果字节必须与实现一致：0 成功、1 重复、2 失败。"""
    assert "| 0 | 成功 |" in doc_text
    assert "| 1 | 重复请求 |" in doc_text
    assert "| 2 | 失败 |" in doc_text
    assert (P.ACK_OK, P.ACK_DUPLICATE, P.ACK_FAILED) == (0, 1, 2)


def test_arm_solution_status_bits_are_documented(doc_text):
    """0x0008 状态字节的结果码必须与实现一致。"""
    from src.algorithms.arm import OUTCOME_NO_TARGET, OUTCOME_OK, OUTCOME_UNREACHABLE
    assert (OUTCOME_OK, OUTCOME_UNREACHABLE, OUTCOME_NO_TARGET) == (0, 1, 2)
    assert "0 成功，1 横向够不着，2 画面里没测到目标" in doc_text


def test_arm_formula_is_documented(doc_text):
    """解算公式必须写在文档里，且与 1.6 节一致。"""
    assert "α = asin(y / l)" in doc_text
    assert "d = x − l·cos α" in doc_text


def test_arm_coordinate_system_is_documented(doc_text):
    """电机坐标系的定义必须写清：原点在车身右侧边中点、y 指向车外。"""
    assert "车身右侧边的中点" in doc_text
    assert "车身本体落在 y < 0 一侧" in doc_text


def test_mcu_must_hold_still_while_waiting_is_mandatory(doc_text):
    """电控在等解算期间必须静止——这条漏了会导致解出的 d 对应一个已失效的姿态。"""
    assert "必须在收到 0x0008 之前保持静止" in doc_text


# ---------- 6. 附录与编码卫生 ----------

def test_appendix_crc_table_agrees_with_implementation(doc_text):
    from src.core.crc import CRC16_TAB, CRC8_TAB

    appendix = doc_text[doc_text.index("附录一"):]
    assert "0x00, 0x5e, 0xbc, 0xe2, 0x61, 0x3f, 0xdd, 0x83" in appendix
    assert "0x35," in appendix
    assert CRC8_TAB[0] == 0x00 and CRC8_TAB[1] == 0x5E and CRC8_TAB[255] == 0x35
    assert "0x0000, 0x1189, 0x2312, 0x329b" in appendix
    assert CRC16_TAB[0] == 0x0000 and CRC16_TAB[1] == 0x1189


def test_appendix_documents_both_init_values(doc_text):
    from src.core.crc import CRC16_INIT, CRC8_INIT
    assert "CRC8_INIT = 0xff" in doc_text and CRC8_INIT == 0xFF
    assert "CRC_INIT = 0xffff" in doc_text and CRC16_INIT == 0xFFFF


def test_doc_has_no_encoding_damage(doc_text):
    """文档必须是无 BOM 的干净 UTF-8：出现私用区字符或替换符即视为损坏。"""
    assert not doc_text.startswith("\ufeff"), "文档带 BOM"
    assert "\ufffd" not in doc_text, "文档含替换字符"
    bad = [c for c in doc_text if 0xE000 <= ord(c) <= 0xF8FF]
    assert not bad, f"文档含 {len(bad)} 个私用区字符（编码损坏）"


def test_old_version_doc_is_marked_superseded(doc_text):
    """旧版本文件保留，但必须标明已被取代，避免有人照着旧版实现。"""
    old = DOC.parent / "protocol-v2.0.0.md"
    if not old.is_file():
        pytest.skip("旧版文档已删除")
    head = old.read_text(encoding="utf-8")[:600]
    assert "V2.1.0" in head and "取代" in head, (
        "旧版文档顶部必须标注已被 V2.1.0 取代")
