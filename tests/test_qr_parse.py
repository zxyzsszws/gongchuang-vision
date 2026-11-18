"""二维码解析与任务计划（结果帧任务码 1）语义测试。

这里覆盖最容易静默出错的一环：第二批在暂存区必须码垛在第一批之上且上下同色，
所以它的靶号要**按颜色配到第一批**，不能沿用第 4 组。
"""
import pytest

from src.core.taskplan import (STATUS_BATCH1_PERMUTATION, STATUS_BATCH2_PERMUTATION,
                               STATUS_COLORS_PAIRABLE, STATUS_USABLE, TaskPlanError,
                               build_task_plan, describe, pair_staging_rings)
from src.algorithms.qr_parse import qr_parse
from src.algorithms.types import QrFormatError


# ---------- 原始解析 ----------

def test_parse_valid():
    raw = qr_parse("452+321+254+312")
    assert raw.batch1_colors == (4, 5, 2)
    assert raw.batch1_positions == (3, 2, 1)
    assert raw.batch2_colors == (2, 5, 4)
    assert raw.batch2_positions == (3, 1, 2)


@pytest.mark.parametrize("bad", ["", "45+321+254+312", "452+321+254",
                                 "452+321+254+317", "abc+321+254+312",
                                 "452+321+254+310", "452+321+254+3125",
                                 "452-321-254-312"])
def test_parse_invalid(bad):
    with pytest.raises(QrFormatError):
        qr_parse(bad)


# ---------- 任务计划装配 ----------

def test_plan_matches_doc_sample():
    """文档 §1.4 黄金样例：452+321+254+312 → 16 字节，status 0x8D。"""
    p = build_task_plan("452+321+254+312")
    assert p.batch1_colors == (4, 5, 2)
    assert p.batch1_rings == (3, 2, 1)
    assert p.batch2_colors == (2, 5, 4)
    assert p.batch2_rings_rough == (3, 1, 2)
    assert p.batch2_rings_staging == (1, 2, 3)
    assert p.status == 0x8D
    assert p.usable
    assert p.to_bytes() == bytes.fromhex(
        "04 05 02 03 02 01 02 05 04 03 01 02 01 02 03 8d")


def test_staging_rings_pair_by_color():
    """颜色 2 在第一批是 1 号靶、5 是 2 号、4 是 3 号 → 暂存区落点 1,2,3。"""
    assert pair_staging_rings((4, 5, 2), (3, 2, 1), (2, 5, 4)) == (1, 2, 3)


def test_staging_rings_general_permutation():
    # 第一批：色1→靶1、色2→靶2、色3→靶3；第二批颜色 (3,1,2) → (3,1,2)
    assert pair_staging_rings((1, 2, 3), (1, 2, 3), (3, 1, 2)) == (3, 1, 2)


def test_duplicate_colors_in_a_batch_rejected():
    """同批内颜色重复 → 配对结果不唯一，必须报错而不是猜一个。"""
    with pytest.raises(TaskPlanError, match="重复"):
        pair_staging_rings((4, 4, 2), (1, 2, 3), (1, 2, 3))
    with pytest.raises(TaskPlanError, match="重复"):
        pair_staging_rings((1, 2, 3), (1, 2, 3), (2, 2, 3))


def test_unpairable_color_makes_plan_unusable():
    """第二批颜色在第一批里找不到 → 不下发（usable=0）。"""
    p = build_task_plan("123+123+456+123")
    assert not p.usable
    assert not (p.status & STATUS_COLORS_PAIRABLE)
    assert not (p.status & STATUS_USABLE)


def test_non_permutation_positions_makes_plan_unusable():
    """第 2 组或第 4 组不是 1,2,3 的排列 → 自检不通过，不下发。"""
    p = build_task_plan("452+321+254+333")
    assert not p.usable
    assert p.status & STATUS_BATCH1_PERMUTATION      # 第 2 组是排列
    assert not (p.status & STATUS_BATCH2_PERMUTATION)  # 第 4 组不是
    assert not (p.status & STATUS_USABLE)

    p2 = build_task_plan("452+322+254+312")
    assert not p2.usable
    assert not (p2.status & STATUS_BATCH1_PERMUTATION)


def test_status_bits_layout():
    assert (STATUS_COLORS_PAIRABLE, STATUS_BATCH1_PERMUTATION,
            STATUS_BATCH2_PERMUTATION, STATUS_USABLE) == (0x01, 0x04, 0x08, 0x80)


def test_to_bytes_is_sixteen_bytes():
    p = build_task_plan("452+321+254+312")
    data = p.to_bytes()
    assert len(data) == 16
    # 前 12 字节就是二维码四组数字原样拼接
    assert data[:12] == bytes([4, 5, 2, 3, 2, 1, 2, 5, 4, 3, 1, 2])


def test_describe_is_human_readable():
    text = describe(build_task_plan("452+321+254+312"))
    assert "0x8D" in text and "批1" in text
