"""任务计划：二维码四组数字 → 0x0002 的 16 字节数据段（对应文档「表 1-7」）。

## 二维码语义

二维码由 4 组三位数组成，用 `+` 连接，例如 `452+321+254+312`：

| 组 | 含义 |
|---|---|
| 第 1 组 | 第一批 3 件物料的颜色与搬运顺序 |
| 第 2 组 | 第一批 3 件物料在**粗加工区与暂存区**的放置位置（两处通用同一靶号） |
| 第 3 组 | 第二批 3 件物料的颜色与搬运顺序 |
| 第 4 组 | 第二批 3 件物料在**粗加工区**的放置位置 |

## 为什么还要单独算「第二批暂存区靶号」

第二批在暂存区必须**码垛在第一批之上，且上下两层颜色一致**。而二维码只保证
「两批颜色集合能配上」，**不保证同靶号同色**，所以第二批在暂存区的落点不能
沿用第 4 组的靶号，必须按颜色去配第一批的那一摞：

    第二批第 i 件 → 在第一批中找同色物料 → 取它的第 2 组靶号

由于**同一批内颜色互不相同**（赛题保证），该查找结果唯一，不需要任何顺序约定。

这个计算放在视觉侧并**明确下发**，而不是让电控自己算：否则两侧各写一遍同一个
判断，只要有一处理解不同就会静默放错（码垛放错颜色照样能放下去，不报任何错）。
"""
from dataclasses import dataclass

from ..algorithms.qr_parse import qr_parse

# 状态字节各位（文档「表 1-7」）
STATUS_COLORS_PAIRABLE = 0x01
STATUS_BATCH1_PERMUTATION = 0x04
STATUS_BATCH2_PERMUTATION = 0x08
STATUS_USABLE = 0x80


class TaskPlanError(ValueError):
    """二维码内容不自洽（颜色配不上、靶号重复），不得下发。"""


@dataclass(frozen=True)
class TaskPlan:
    """一份可直接打包成 0x0002 的完整任务计划。"""

    batch1_colors: tuple
    batch1_rings: tuple
    batch2_colors: tuple
    batch2_rings_rough: tuple
    batch2_rings_staging: tuple
    status: int

    @property
    def usable(self) -> bool:
        return bool(self.status & STATUS_USABLE)

    def to_bytes(self) -> bytes:
        """打包为 16 字节数据段（文档「表 1-7」的字段顺序）。"""
        return bytes(self.batch1_colors + self.batch1_rings
                     + self.batch2_colors + self.batch2_rings_rough
                     + self.batch2_rings_staging + (self.status,))


def _is_permutation(values, n):
    return sorted(values) == list(range(1, n + 1))


def pair_staging_rings(batch1_colors, batch1_rings, batch2_colors):
    """按颜色把第二批配到第一批的靶号上（码垛落点）。

    返回与 ``batch2_colors`` 等长的靶号元组。同一批内颜色重复时**结果不唯一**，
    但赛题保证不重复，这里一旦发现重复直接抛错，不猜。
    """
    if len(set(batch1_colors)) != len(batch1_colors):
        raise TaskPlanError(f"第一批颜色有重复 {batch1_colors}，配色落点不唯一")
    if len(set(batch2_colors)) != len(batch2_colors):
        raise TaskPlanError(f"第二批颜色有重复 {batch2_colors}，配色落点不唯一")
    out = []
    for color in batch2_colors:
        if color not in batch1_colors:
            raise TaskPlanError(
                f"第二批颜色 {color} 在第一批 {batch1_colors} 里找不到，无法码垛")
        out.append(batch1_rings[batch1_colors.index(color)])
    return tuple(out)


def build_task_plan(text: str) -> TaskPlan:
    """解析二维码文本并装配任务计划；内容不自洽时抛 ``TaskPlanError``。"""
    raw = qr_parse(text)

    status = 0
    if _is_permutation(raw.batch1_positions, len(raw.batch1_positions)):
        status |= STATUS_BATCH1_PERMUTATION
    if _is_permutation(raw.batch2_positions, len(raw.batch2_positions)):
        status |= STATUS_BATCH2_PERMUTATION

    try:
        staging = pair_staging_rings(raw.batch1_colors, raw.batch1_positions,
                                     raw.batch2_colors)
    except TaskPlanError:
        # 配不上：不下发（usable=0 由调用方判断）
        return TaskPlan(raw.batch1_colors, raw.batch1_positions,
                        raw.batch2_colors, raw.batch2_positions,
                        tuple(0 for _ in raw.batch2_colors), status)

    status |= STATUS_COLORS_PAIRABLE
    # 三项自检全过才允许电控按本帧规划
    need = STATUS_COLORS_PAIRABLE | STATUS_BATCH1_PERMUTATION | STATUS_BATCH2_PERMUTATION
    if status & need == need:
        status |= STATUS_USABLE

    return TaskPlan(raw.batch1_colors, raw.batch1_positions,
                    raw.batch2_colors, raw.batch2_positions,
                    staging, status)


def describe(plan: TaskPlan) -> str:
    """给日志用的一行摘要（不参与协议）。"""
    return (f"批1 色{plan.batch1_colors}→靶{plan.batch1_rings} | "
            f"批2 色{plan.batch2_colors}→粗{plan.batch2_rings_rough} "
            f"暂{plan.batch2_rings_staging} | status=0x{plan.status:02X}")
