"""场地图：两组参考靶、转盘、起点、码垛高度（对应文档附录三）。

坐标与角度约定见 `docs/protocol-v2.1.0.md` 附录二：
原点为开赛停车位上机械臂底座中心在地面的投影，x 轴正方向为开赛时车头方向，
y 轴正方向为车头左侧，角度俯视逆时针为正，单位 mm。
"""
from dataclasses import dataclass, field

import yaml

from ..core.protocol import REGION_MATERIAL, REGION_ROUGH, REGION_STAGING, REGION_START
from .types import Pose, RingObs

# 去向提示取值 → 场地图中的参考靶组名（文档「表 1-10」）
REGION_TO_GROUP = {REGION_ROUGH: "rough", REGION_STAGING: "staging"}


@dataclass(frozen=True)
class TargetField:
    """一个参考靶：编号 + 场地圆心 + 最外圈半径 + 同心圈数。"""

    label: int
    center: tuple
    radius: float
    rings: int = 1

    @property
    def obs(self) -> RingObs:
        return RingObs(self.center, self.radius, self.label, kind="target")


@dataclass(frozen=True)
class TargetGroup:
    """一处的三个参考靶，以及车在这一处的标称入场位姿。

    ``entry_pose`` 不是「电控的推算值」，而是**赛前测定的标称位姿**：车停到该区域
    正常作业位置时，机械臂底座中心大致在哪、车头大致朝哪。它的唯一用途是消解
    三个共线靶的 180° 镜像歧义（见 ``algorithms/localize.py``）——两个候选解中
    取与标称位姿更接近的那个。车实际停偏几十毫米、转头几度都不影响这个判断，
    因为两个候选解相差 180°。
    """

    targets: tuple
    entry_pose: Pose = Pose(0.0, 0.0, 0.0)

    def __iter__(self):
        return iter(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.targets[i]


@dataclass(frozen=True)
class TurntableField:
    """转盘的场地信息（视觉只在车身坐标系下报它，这里用于合成场景与校核）。"""

    center: tuple
    station_radius: float
    station_count: int = 3


@dataclass(frozen=True)
class FieldMap:
    """整张场地图。``groups`` 的键为 ``"rough"`` / ``"staging"``。"""

    groups: dict = field(default_factory=dict)
    turntable: TurntableField | None = None
    start: Pose = Pose(0.0, 0.0, 0.0)
    layer_height_mm: float = 0.0
    # 各区域（去向提示取值）的标称停车位姿。只在 sim 里用来生成一条合理的
    # 行驶路线；真实系统里这套路径由电控自己规划，视觉不参与。
    region_poses: dict = field(default_factory=dict)

    def group(self, name):
        """按名字取一组靶；不存在时抛 KeyError（比返回空列表更早暴露配置错误）。"""
        return self.groups[name]

    def pose_for_region(self, region):
        """该区域的标称停车位姿；未配置时退回 ``start``。"""
        if region in self.region_poses:
            return self.region_poses[region]
        if region == REGION_START:
            return self.start
        return None

    def group_for_region(self, region):
        """按去向提示取值取靶组；该区域没有参考靶（原料区/起点）时返回 None。"""
        return self.groups.get(REGION_TO_GROUP.get(region))

    def targets(self, name):
        """该组三个靶按编号升序返回。"""
        return sorted(self.group(name).targets, key=lambda t: t.label)

    def entry_pose_for_region(self, region):
        """该区域的标称入场位姿；无参考靶的区域返回 None。"""
        group = self.group_for_region(region)
        return group.entry_pose if group is not None else None

    def target(self, name, label):
        """按编号取单个靶；不存在时抛 KeyError。"""
        for t in self.group(name).targets:
            if t.label == label:
                return t
        raise KeyError(f"{name} 组没有编号 {label} 的靶")


def _parse_pose(raw):
    raw = raw or {}
    return Pose(float(raw.get("x", 0.0)), float(raw.get("y", 0.0)),
                float(raw.get("heading_deg", 0.0)))


def _parse_group(name, raw):
    targets = []
    for item in raw.get("targets") or []:
        targets.append(TargetField(
            label=int(item["label"]),
            center=(float(item["center"][0]), float(item["center"][1])),
            radius=float(item["radius"]),
            rings=int(item.get("rings", 1)),
        ))
    if not targets:
        raise ValueError(f"参考靶组 {name} 没有任何靶")
    labels = [t.label for t in targets]
    if len(set(labels)) != len(labels):
        raise ValueError(f"参考靶组 {name} 的编号有重复：{labels}")

    # 三个靶必须近共线，否则「沿连线量方向」这个前提不成立，位姿解出来是错的
    if len(targets) >= 3:
        pts = [t.center for t in sorted(targets, key=lambda t: t.label)]
        _require_collinear(name, pts)

    return TargetGroup(tuple(targets), _parse_pose(raw.get("entry_pose")))


def _require_collinear(name, pts, tol_mm=15.0):
    """检查首尾连线到中间点的垂距；超过 tol_mm 说明三点明显不共线。"""
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    span = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if span < 1e-6:
        raise ValueError(f"参考靶组 {name} 的首尾靶重合，无法定出方向")
    for x, y in pts[1:-1]:
        # 点到直线距离
        d = abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / span
        if d > tol_mm:
            raise ValueError(
                f"参考靶组 {name} 的三个靶不在一条直线上（垂距 {d:.1f}mm > "
                f"{tol_mm}mm）。朝向靠三点连线的方向解算，必须共线摆放。")


def load_field_map(path: str) -> FieldMap:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    groups = {name: _parse_group(name, body)
              for name, body in (raw.get("reference_groups") or {}).items()}
    if not groups:
        raise ValueError(f"{path}: 缺少 reference_groups，视觉无法解算位姿")

    tt = raw.get("turntable")
    turntable = None
    if tt:
        turntable = TurntableField(
            center=(float(tt["center"][0]), float(tt["center"][1])),
            station_radius=float(tt["station_radius"]),
            station_count=int(tt.get("station_count", 3)),
        )

    start = _parse_pose((raw.get("start") or {}).get("pose"))
    layer = float((raw.get("palletizing") or {}).get("layer_height_mm", 0.0))

    # 各区域标称停位（sim 生成行驶路线用；真实系统由电控自己规划）
    keymap = {"material": REGION_MATERIAL, "rough": REGION_ROUGH,
              "staging": REGION_STAGING}
    region_poses = {}
    for name, pose in (raw.get("region_poses") or {}).items():
        if name not in keymap:
            raise ValueError(
                f"{path}: region_poses 里有未知区域 {name!r}，"
                f"可选 {sorted(keymap)}")
        region_poses[keymap[name]] = _parse_pose(pose)

    return FieldMap(groups=groups, turntable=turntable, start=start,
                    layer_height_mm=layer, region_poses=region_poses)
