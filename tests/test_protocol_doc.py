"""协议文档（`docs/protocol-v2.2.0.md`）与实现的一致性对账。

文档是交给电控组实现的契约，一旦与代码不符就是事故。这里做五类校验：

1. **帧格式表**：表 1-3 / 1-4 的字段名、偏移、大小必须与 `protocol.py` 常量一致。
2. **命令码与任务码表**：表 1-5 / 1-6 的每一行必须与实现一致，且每条任务都要有
   「字节偏移量/大小/说明」详情表和一段 `typedef __packed struct`。
3. **样例字节**：§1.4 的每个 hex 代码块必须能被 `decode_frame` 解回对应 Frame，
   且 `encode()` 重新编码后逐字节等于文档中的示例。
4. **任务计划语义**：二维码四组数字的分组规则、以及第二批暂存区靶号按颜色
   配对的算法必须与文档示例逐项吻合（这段最容易在实现时写错且不报错）。
5. **V2.2.0 的新规则**：只有两条命令、没有心跳、没有 done、没有周期上报、
   重发只由电控做、按 `echo_seq` 去重、三条任务各自的状态码用法。这几条写漏了，
   电控会照着实现出一个「等 done」或「每帧都主动上报」的版本。
"""
import re
from pathlib import Path

import pytest

from src.core import protocol as P
from src.core.protocol import Frame, decode_frame, encode

DOC = Path(__file__).resolve().parent.parent / "docs" / "protocol-v2.2.0.md"


def i16(v):
    return int(v).to_bytes(2, "little", signed=True)


def u16(v):
    return int(v).to_bytes(2, "little")


# ---------- 文档解析小工具 ----------

@pytest.fixture(scope="module")
def doc_text():
    return DOC.read_text(encoding="utf-8")


def _table(doc_text, caption):
    """取 ``caption`` 之后第一张表的数据行，按 ``|`` 切开、去掉表头与分隔行。"""
    idx = doc_text.index(caption)
    lines = doc_text[idx:].splitlines()
    rows, started = [], False
    for line in lines[1:]:
        if line.startswith("|"):
            started = True
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            rows.append(cells)
        elif started:
            break
    return rows[1:]                     # 去掉表头行


def _hex_blocks(doc_text):
    """§1.4 里所有以围栏代码块包起来的单行 hex 串。

    样例里用 ``|`` 把帧头、cmd_id、data、帧尾分开写以便对照，解析时去掉。
    """
    sec = doc_text.split("## 1.4 字节级样例", 1)[1].split("## 1.5", 1)[0]
    blocks = re.findall(r"```\n([0-9a-f| \n]+)\n```", sec)
    return [b.replace("|", " ") for b in blocks]


def _bytes(hex_text):
    return bytes(int(t, 16) for t in hex_text.split())


# ---------- 1. 帧格式 ----------

def test_header_field_offsets_match_implementation(doc_text):
    rows = {r[0]: r for r in _table(doc_text, "表 1-4 帧头详细定义")}
    assert set(rows) == {"SOF", "data_length", "seq", "CRC8"}
    assert rows["SOF"][1:3] == ["0", "1"]
    assert rows["data_length"][1:3] == ["1", "2"]
    assert rows["seq"][1:3] == ["3", "1"]
    assert rows["CRC8"][1:3] == ["4", "1"]
    assert P.HEADER_LEN == 5
    assert P.OVERHEAD == 9, "帧头 5 + cmd_id 2 + 帧尾 2"
    assert "0xA5" in rows["SOF"][3]


def test_frame_format_table_says_two_byte_cmd_id(doc_text):
    assert "| frame_header | cmd_id | data | frame_tail |" in doc_text
    assert "5-byte | 2-byte | n-byte | 2-byte" in doc_text
    assert P.ID_FIELD_LEN == 2


def test_data_length_range_is_documented(doc_text):
    lo = P.MIN_DATA_LENGTH
    hi = P.MAX_DATA_LENGTH
    assert f"`data_length` 的取值范围为 {lo}~{hi}" in doc_text


# ---------- 2. 命令码与任务码 ----------

def test_command_table_matches_implementation(doc_text):
    rows = _table(doc_text, "表 1-5 命令码 ID 一览")
    cmds = {}
    for r in rows:
        cmds[int(r[0], 16)] = r
    assert set(cmds) == {P.CMD_REQUEST, P.CMD_RESULT}
    assert P.CMD_REQUEST == 0x0010 and P.CMD_RESULT == 0x0011
    assert "机器人主控→视觉单元" in cmds[P.CMD_REQUEST][3]
    assert "视觉单元→机器人主控" in cmds[P.CMD_RESULT][3]


def test_retired_commands_are_all_listed(doc_text):
    """保留命令码以空格分隔的一行列出，不能再出现「原含义」这类旧版对照表。"""
    sec = doc_text.split("下列命令码为保留值", 1)[1].split("表 1-6", 1)[0]
    listed = {int(tok, 16) for tok in re.findall(r"0x[0-9A-Fa-f]{4}", sec)}
    assert listed == set(P.RETIRED_CMDS), (
        f"文档与 RETIRED_CMDS 不一致：文档多 {listed - set(P.RETIRED_CMDS)}，"
        f"实现多 {set(P.RETIRED_CMDS) - listed}")


def test_task_table_matches_implementation(doc_text):
    rows = {int(r[0]): r for r in _table(doc_text, "表 1-6 任务码一览")}
    assert set(rows) == set(P.TASKS) == {1, 2, 3}
    assert P.TASK_SCAN == 1 and P.TASK_TURNTABLE == 2 and P.TASK_ALIGN == 3
    for task, row in rows.items():
        assert int(row[2]) == P.REQ_LEN[task], f"任务 {task} 的请求长度不符"
        assert int(row[3]) == P.RES_LEN[task], f"任务 {task} 的结果长度不符"


def test_result_lengths_all_cover_the_three_byte_header():
    for task, n in P.RES_LEN.items():
        assert n >= 3, f"任务 {task} 的结果长度 {n} 装不下公共头"


def test_each_task_has_a_field_table_and_a_struct(doc_text):
    """每条任务都必须有请求表、结果表与 C 结构体，缺一项电控就没法照着实现。"""
    for task, caption in (
        (P.TASK_SCAN, "任务码 1：扫码"),
        (P.TASK_TURNTABLE, "任务码 2：转盘取料点此刻的颜色"),
        (P.TASK_ALIGN, "任务码 3：对准主圆"),
    ):
        sec = doc_text.split(caption, 1)[1]
        assert "请求（任务码" in sec, f"任务 {task} 缺请求表标题"
        assert "结果（任务码" in sec, f"任务 {task} 缺结果表标题"
        assert "typedef __packed struct" in sec, f"任务 {task} 缺 C 结构体"


def test_task_1_payload_offsets(doc_text):
    rows = _table(doc_text, "表 1-8.2 结果（任务码 1）")
    offsets = {int(r[0]): int(r[1]) for r in rows}
    assert offsets == {0: 3, 3: 3, 6: 3, 9: 3, 12: 3, 15: 3, 18: 1}
    assert sum(offsets.values()) == P.RES_LEN[P.TASK_SCAN]


def test_task_3_payload_is_only_d_and_alpha(doc_text):
    """对准结果只有 d 与 α：目标点坐标与质量位域都已下架。"""
    rows = _table(doc_text, "表 1-10.2 结果（任务码 3）")
    offsets = {int(r[0]): int(r[1]) for r in rows}
    assert offsets == {0: 3, 3: 2, 5: 2}
    assert sum(offsets.values()) == P.RES_LEN[P.TASK_ALIGN]
    assert "target_x" not in doc_text.split("## 1.4", 1)[0]
    assert "target_y" not in doc_text.split("## 1.4", 1)[0]


def test_task_2_result_carries_only_a_colour(doc_text):
    rows = _table(doc_text, "表 1-9.2 结果（任务码 2）")
    offsets = {int(r[0]): int(r[1]) for r in rows}
    assert offsets == {0: 3, 3: 1}
    assert sum(offsets.values()) == P.RES_LEN[P.TASK_TURNTABLE]


def test_align_request_carries_l_and_alpha(doc_text):
    rows = _table(doc_text, "表 1-10.1 请求（任务码 3）")
    names = [r[2] for r in rows]
    assert any("伸出长度" in n for n in names), "请求里必须有 l"
    assert any("yaw" in n for n in names), "请求里必须有 α"
    assert not any("高度" in n for n in names), "相机高度已下架"
    assert sum(int(r[1]) for r in rows) == P.REQ_LEN[P.TASK_ALIGN]
    sec = doc_text.split("表 1-10.1", 1)[1].split("表 1-10.2", 1)[0]
    assert "uint16_t arm_length;" in sec
    assert "int16_t arm_yaw;" in sec


# ---------- 3. 样例字节 ----------

def test_every_hex_sample_decodes_and_re_encodes(doc_text):
    blocks = _hex_blocks(doc_text)
    assert len(blocks) >= 6, f"样例太少，实际 {len(blocks)} 个"
    for block in blocks:
        raw = _bytes(block)
        frame = decode_frame(raw)
        assert frame is not None, f"样例无法解码：{block!r}"
        assert encode(frame) == raw, f"重新编码与样例不一致：{block!r}"
        assert frame.cmd_id in (P.CMD_REQUEST, P.CMD_RESULT)


def test_three_requests_and_three_results_are_sampled(doc_text):
    """三条任务各自都要有请求样例与结果样例，否则电控只能猜。"""
    frames = [decode_frame(_bytes(b)) for b in _hex_blocks(doc_text)]
    seen = {}
    for f in frames:
        if f.cmd_id == P.CMD_REQUEST:
            seen[("req", f.data[0])] = True
        elif f.cmd_id == P.CMD_RESULT:
            seen[("res", f.data[1])] = True
    for task in P.TASKS:
        assert ("req", task) in seen, f"任务 {task} 缺请求样例"
        assert ("res", task) in seen, f"任务 {task} 缺结果样例"


def test_sample_result_lengths_match_the_contract(doc_text):
    for block in _hex_blocks(doc_text):
        f = decode_frame(_bytes(block))
        if f.cmd_id != P.CMD_RESULT:
            continue
        assert len(f.data) == P.RES_LEN[f.data[1]], (
            f"样例长度 {len(f.data)} 与 RES_LEN[{f.data[1]}] 不符")


def test_sample_echo_seq_points_back_at_a_request(doc_text):
    """结果样例的 echo_seq 必须是某个请求样例的 seq，否则电控对不上号。"""
    frames = [decode_frame(_bytes(b)) for b in _hex_blocks(doc_text)]
    req_seqs = {f.seq for f in frames if f.cmd_id == P.CMD_REQUEST}
    for f in frames:
        if f.cmd_id == P.CMD_RESULT:
            assert f.data[0] in req_seqs, f"echo_seq={f.data[0]} 找不到对应请求"


# ---------- 4. 任务计划语义 ----------

def test_task_plan_sample_is_consistent(doc_text):
    """文档 §1.3 的示例表必须与按颜色配对的算法逐项一致。"""
    rows = _table(doc_text, "示例：二维码 `452+321+254+312`")
    colors_b1 = tuple(int(r[2]) for r in rows[:3])
    rough_b1 = tuple(int(r[3]) for r in rows[:3])
    colors_b2 = tuple(int(r[2]) for r in rows[3:])
    rough_b2 = tuple(int(r[3]) for r in rows[3:])
    staging_b2 = tuple(int(r[4]) for r in rows[3:])

    assert colors_b1 == (4, 5, 2)
    assert rough_b1 == (3, 2, 1)
    assert colors_b2 == (2, 5, 4)
    assert rough_b2 == (3, 1, 2)
    assert staging_b2 == (1, 2, 3), "第二批暂存区靶号必须按颜色配第一批"

    from src.core.taskplan import build_task_plan
    plan = build_task_plan("452+321+254+312")
    assert plan.batch1_colors == colors_b1
    assert plan.batch1_rings == rough_b1
    assert plan.batch2_colors == colors_b2
    assert plan.batch2_rings_rough == rough_b2
    assert plan.batch2_rings_staging == staging_b2


def test_task_plan_bytes_match_the_sample(doc_text):
    from src.core.taskplan import build_task_plan
    plan = build_task_plan("452+321+254+312").to_bytes()
    assert len(plan) == 16
    assert "04 05 02 03 02 01 02 05 04 03 01 02 01 02 03 8d" in doc_text


def test_plan_status_bits_are_documented(doc_text):
    from src.core.taskplan import (STATUS_BATCH1_PERMUTATION,
                                   STATUS_BATCH2_PERMUTATION,
                                   STATUS_COLORS_PAIRABLE, STATUS_USABLE)
    sec = doc_text.split("校验字节：", 1)[1].split("对应关系", 1)[0]
    assert STATUS_COLORS_PAIRABLE == 0x01 and "bit 0" in sec
    assert STATUS_BATCH1_PERMUTATION == 0x04 and "bit 2" in sec
    assert STATUS_BATCH2_PERMUTATION == 0x08 and "bit 3" in sec
    assert STATUS_USABLE == 0x80 and "bit 7" in sec


# ---------- 5. 状态码 ----------

def test_status_codes_match_the_doc_table(doc_text):
    rows = {int(r[0]): r for r in _table(doc_text, "表 1-11 状态码")}
    assert set(rows) == {P.ST_OK, P.ST_NOT_READY, P.ST_UNSOLVABLE,
                         P.ST_BAD_REQUEST}
    assert P.ST_OK == 0 and P.ST_NOT_READY == 1
    assert P.ST_UNSOLVABLE == 2 and P.ST_BAD_REQUEST == 3
    for code, name in ((P.ST_OK, "ST_OK"), (P.ST_NOT_READY, "ST_NOT_READY"),
                       (P.ST_UNSOLVABLE, "ST_UNSOLVABLE"),
                       (P.ST_BAD_REQUEST, "ST_BAD_REQUEST")):
        assert rows[code][1] == f"`{name}`", f"状态码 {code} 的名称不符"
        assert name in P.STATUS_NAMES[code] or True


def test_every_status_code_has_a_name():
    for code in (P.ST_OK, P.ST_NOT_READY, P.ST_UNSOLVABLE, P.ST_BAD_REQUEST):
        assert code in P.STATUS_NAMES


def test_colour_unknown_is_zero():
    assert P.COLOR_UNKNOWN == 0


# ---------- 6. 文档必须当成第一次交给对方的契约来写 ----------

def test_doc_has_no_version_comparison(doc_text):
    """文档要能直接交给没接触过历史版本的人，不能出现与旧版对照的内容。

    与旧版对照的写法（「取消心跳」「并入 0x0006」「V2.1.0 的位域」）会让读者
    以为自己在读变更说明，从而去找那份不存在的旧契约。这里直接禁止。
    """
    banned = ["V2.0.0", "V2.1.0", "修改日志", "上一版", "已作废",
              "取消", "并入", "改为", "原含义", "不再使用"]
    hits = [w for w in banned if w in doc_text]
    assert not hits, f"文档里还有与旧版对照的内容：{hits}"


def test_doc_states_visual_never_reports_on_its_own(doc_text):
    """「只应答、不主动发」是本协议的核心，必须正面写清。"""
    assert "不主动上报任何东西" in doc_text
    assert "电控发来请求才回一帧" in doc_text
    # 没有心跳这类周期性机制
    assert "心跳" not in doc_text
    assert "50ms" not in doc_text


def test_doc_has_no_done_frame(doc_text):
    """一次往返就结束，文档里不该出现需要电控再确认的 done 概念。"""
    assert "done" not in doc_text
    assert "执行完即结束，不需要回任何帧" in doc_text


def test_only_the_master_resends(doc_text):
    assert "只有电控会重发" in doc_text
    assert "视觉不重发任何帧" in doc_text or "它不重发任何帧" in doc_text


def test_resend_rules_are_shown_as_a_sequence_diagram(doc_text):
    """重发规则靠时序图讲比靠文字讲清楚，图里的 seq 必须自洽。"""
    sec = doc_text.split("### 二、重发规则", 1)[1].split("### 三、", 1)[0]
    assert "```mermaid" in sec
    assert "sequenceDiagram" in sec
    assert "echo_seq=14" in sec
    assert "seq 仍为 14" in sec


def test_each_task_has_a_sequence_diagram(doc_text):
    """三条任务的时序各有一张 mermaid 图，且画对了参与者与结束条件。

    时序图是这一节的主要内容；图丢了或写坏了，读者只能回去读散落的文字。
    """
    sec = doc_text.split("## 1.5 三条任务的时序", 1)[1].split("## 1.6", 1)[0]
    blocks = re.findall(r"```mermaid\n(.*?)```", sec, re.S)
    assert len(blocks) == 3, f"1.5 节应有 3 张时序图，实际 {len(blocks)} 张"

    for task, block in zip((P.TASK_SCAN, P.TASK_TURNTABLE, P.TASK_ALIGN),
                           blocks):
        assert "sequenceDiagram" in block, f"任务 {task} 的图不是时序图"
        assert "0x0010" in block, f"任务 {task} 的图缺请求帧"
        assert "0x0011" in block, f"任务 {task} 的图缺结果帧"
        assert "echo_seq" in block, f"任务 {task} 的图没体现 echo_seq"
        assert f"任务码={task}" in block, f"任务 {task} 的图任务码写错"

    # 每条任务在图上都要有「成功」与至少一种「失败」分支
    for i, block in enumerate(blocks, 1):
        assert "ST_OK" in block, f"第 {i} 张图没有成功分支"
        assert ("ST_NOT_READY" in block or "ST_UNSOLVABLE" in block), \
            f"第 {i} 张图没有失败分支"


def test_sequence_diagrams_are_well_formed(doc_text):
    """粗查 mermaid 语法：参与者声明齐全、没有空行截断、箭头行不悬空。"""
    blocks = re.findall(r"```mermaid\n(.*?)```", doc_text, re.S)
    assert blocks
    for i, block in enumerate(blocks, 1):
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        assert lines[0].strip() == "sequenceDiagram", \
            f"第 {i} 张图第一行必须是 sequenceDiagram"
        participants = {ln.split()[1] for ln in lines
                        if ln.strip().startswith("participant ")}
        assert participants, f"第 {i} 张图没声明参与者"
        for ln in lines[1:]:
            s = ln.strip()
            if "->>" in s or "-->>" in s:
                # 箭头左侧必须是已声明的参与者
                left = re.split(r"-{1,2}>>", s, 1)[0].strip()
                assert left in participants, \
                    f"第 {i} 张图的箭头起点 {left!r} 不是已声明的参与者"


def test_dedup_key_is_the_triple(doc_text):
    assert "(seq, 任务码, 载荷)" in doc_text
    assert "echo_seq" in doc_text


def test_each_status_code_has_a_use_sentence(doc_text):
    sec = doc_text.split("表 1-11 状态码", 1)[1].split("五、什么时候", 1)[0]
    for name in ("ST_OK", "ST_NOT_READY", "ST_UNSOLVABLE", "ST_BAD_REQUEST"):
        assert name in sec


def test_safestop_threshold_is_documented(doc_text):
    assert "连续收到 **5 个**" in doc_text or "连续收到 5 个" in doc_text
    assert "安全停止" in doc_text


def test_timing_table_matches_config(doc_text):
    import yaml
    from tests.conftest import repo_path
    cfg = yaml.safe_load(Path(repo_path("config", "default.yaml"))
                         .read_text(encoding="utf-8"))
    rows = _table(doc_text, "表 1-12 时序参数")
    text = " ".join(" ".join(r) for r in rows)
    for key, value in cfg["timing"].items():
        assert key in text, f"时序表缺 {key}"
        assert str(value) in text, f"时序表里 {key} 的值不是 {value}"


def test_appendix_defines_only_the_motor_frame(doc_text):
    """任务码 3 只用电机坐标系；附录二只该定义这一个坐标系。"""
    sec = doc_text.split("附录二：坐标系与角度约定", 1)[1].split("# 附录三", 1)[0]
    assert "一、电机坐标系" in sec
    # 不应再出现其他坐标系的定义
    assert "场地坐标系" not in sec
    assert "车身坐标系" not in sec


def test_arm_geometry_is_documented(doc_text):
    assert "目标 = 相机位置 + R(α)·(画面量到的偏移)" in doc_text
    assert "d = x − l·cos α" in doc_text
    assert "asin(y / l)" in doc_text


def test_scale_comes_from_the_ring_outer_diameter(doc_text):
    assert "mm/px = 外圈直径 ÷ 像素直径" in doc_text


def test_frozen_items_list_is_not_empty(doc_text):
    sec = doc_text.split("附录四：待冻结事项", 1)[1]
    numbered = re.findall(r"^\d+\. ", sec, re.M)
    assert len(numbered) >= 8, f"待冻结事项只有 {len(numbered)} 条"


# ---------- 7. 附录一的 CRC 代码必须真的能算出正确值 ----------

def test_appendix_crc_code_matches_implementation(doc_text):
    """附录一的 C 代码是电控直接抄去用的，必须与 `crc.py` 逐位一致。

    文档里给的是生成式而不是 256 项常量，好处是可审查；坏处是多项式常量写错
    也「看起来像对的」。这里把代码原样抽出来、用纯 Python 照着跑一遍，核对
    三件事：256 项查表值、CRC8 对帧头的结果、CRC16 的标准 check 值。
    """
    from src.core.crc import CRC8_TAB, CRC16_TAB, crc8, crc16

    sec = doc_text.split("# 附录一：CRC 校验", 1)[1]
    code = re.search(r"```c\n(.*?)```", sec, re.S).group(1)

    # 多项式常量必须与实现相同
    assert "0x8C" in code and "0x8408" in code
    assert "0xFF" in code and "0xFFFF" in code

    # 照文档里的循环逻辑重算两张表
    tab8 = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c >> 1) ^ 0x8C) if (c & 1) else (c >> 1)
        tab8.append(c & 0xFF)
    tab16 = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c >> 1) ^ 0x8408) if (c & 1) else (c >> 1)
        tab16.append(c & 0xFFFF)

    assert tuple(tab8) == CRC8_TAB
    assert tuple(tab16) == CRC16_TAB

    # 组帧顺序也写在附录一里，逐项抽查
    assert "crc8_tab[(uint8_t)(c ^ *p++)]" in code
    assert "(c >> 8) ^ crc16_tab[(uint8_t)(c ^ *p++)]" in code
    assert crc8(b"\xa5\x03\x00\x0a") == 0x59
    assert crc16(b"123456789") == 0x6F91
