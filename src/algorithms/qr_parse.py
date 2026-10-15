import re

from .types import TaskPlan, QrFormatError

_PATTERN = re.compile(r"^\d{3}\+\d{3}\+\d{3}\+\d{3}$")


def _digits(group: str, lo: int, hi: int) -> tuple:
    out = tuple(int(ch) for ch in group)
    if any(not lo <= d <= hi for d in out):
        raise QrFormatError(group)
    return out


def qr_parse(text: str) -> TaskPlan:
    """解析竞赛二维码文本为 TaskPlan。

    语义（飞书文档 1.1）：``452+321+254+312`` →
    第1组=第一批搬运颜色序、第2组=第一批放置位置、
    第3组=第二批颜色序、第4组=第二批放置位置。
    颜色数字 1-6（红黄蓝绿黑浅蓝）；位置数字 1-3=环编号。
    """
    if not _PATTERN.match(text):
        raise QrFormatError(text)
    g1, g2, g3, g4 = text.split("+")
    return TaskPlan(
        batch1_colors=_digits(g1, 1, 6),
        batch1_positions=_digits(g2, 1, 3),
        batch2_colors=_digits(g3, 1, 6),
        batch2_positions=_digits(g4, 1, 3),
    )
