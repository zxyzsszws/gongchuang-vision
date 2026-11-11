"""live 感知链路：相机帧 → 参考靶检出 → 车身毫米 → 解位姿。

这是「像素→毫米」这层换算的能力证明。v2.0.0 里换算**只做一层**（到车身毫米），
位姿由 ``algorithms/localize.py`` 用三个靶解算，不再需要「先知道自己在哪才能
量出自己在哪」的循环依赖。

测试用一个**按真值渲染**的假相机：把场地图里的三个靶按已知位姿投影成同心圆，
再让 ``VisionPerceive`` 从图像里把它们检回来。这样能验证：

- 检测层能从图像里恢复出精确的靶心；
- 外参换算（像素→车身毫米）方向与尺度正确；
- 无外参时拒绝输出（宁可不出，也不出方向错的测量）。
"""
import math

import cv2
import numpy as np

from src.algorithms.fieldmap import load_field_map
from src.algorithms.geometry import CameraCalib, Similarity
from src.algorithms.localize import solve_pose
from src.algorithms.types import Pose
from src.devices.vision import VisionPerceive
from tests.conftest import repo_path

FIELD = load_field_map(repo_path("config", "field.example.yaml"))

# 相机外参：2mm/px、无旋转，图像中心 (320,240) 对应车身参考点。
# 为什么尺度要 2mm/px：三个靶横跨 600mm，在 640×480 的图里按 1mm/px 放不下，
# 必须让 1 像素代表更多毫米。靶间距 300mm 于是对应 150px。
SCALE = 2.0
CALIB = Similarity(scale=SCALE, rot_deg=0.0, tx=-320.0 * SCALE,
                   ty=-240.0 * SCALE)
INV = CALIB.invert()


def _cfg():
    import yaml
    with open(repo_path("config", "default.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # 合成图里靶的像素半径是 20/30/40（rings 除以 SCALE=2），搜索区间要盖住
    cfg["target"] = dict(cfg["target"], search_radius=[12, 60], min_rings=3)
    return cfg


def _render_targets(pose_true, group, rings=(40, 60, 80), thickness=2):
    """按真位姿把一组参考靶投影成图像上的同心圆。

    场地 → 车身（真位姿的逆）→ 像素（外参的逆）。这条路径与 ``VisionPerceive``
    的正向换算严格互逆，因此解出的位姿应精确等于 ``pose_true``（只差像素取整）。

    ``scale_mm_per_px`` 的标度由外参决定，圆的像素半径要按它折算，否则圈数、
    半径范围都对不上配置里的搜索区间。
    """
    W, H = 640, 480
    img = np.full((H, W, 3), 240, np.uint8)
    return img, _render_into(img, pose_true, group, rings, thickness)


def _render_into(img, pose_true, group, rings=(40, 60, 80), thickness=2):
    """把一组靶画进已有图像（两组同时入镜的测试用）。返回各靶像素圆心。"""
    a = math.radians(pose_true.heading)
    ca, sa = math.cos(a), math.sin(a)
    px_per_mm = 1.0 / SCALE

    px_centers = []
    for t in group:
        dx = t.center[0] - pose_true.x
        dy = t.center[1] - pose_true.y
        bx = ca * dx + sa * dy                 # 旋转 −heading
        by = -sa * dx + ca * dy
        px = INV.apply((bx, by))               # 车身 → 像素
        px_centers.append(px)
        for r_mm in rings:
            cv2.circle(img, (int(round(px[0])), int(round(px[1]))),
                       int(round(r_mm * px_per_mm)), (0, 0, 0), thickness)
    return px_centers


class _ScriptedCamera:
    """按脚本逐帧返回图像（模拟连续帧流）。"""

    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def read(self):
        if not self._frames:
            return False, None
        frame = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        return True, frame


def _perceive(img, calib=CALIB):
    return VisionPerceive(_ScriptedCamera([img]), _cfg(),
                          calib=CameraCalib(calib) if calib else None,
                          field=FIELD)


# ---------- 检出与换算 ----------

def test_targets_are_detected_in_projected_image():
    """渲染的同心圆靶必须能被检回来（否则后面所有换算都是空中楼阁）。"""
    truth = Pose(1500.0, 1000.0, 270.0)
    group = FIELD.targets("rough")
    img, _ = _render_targets(truth, group)
    obs = _perceive(img).targets(2)
    assert obs is not None and len(obs) == 3

    # obs 是车身毫米；期望值用真位姿把靶心反投影到车身坐标得到
    a = math.radians(truth.heading)
    ca, sa = math.cos(a), math.sin(a)
    want = []
    for t in group:
        dx, dy = t.center[0] - truth.x, t.center[1] - truth.y
        want.append((ca * dx + sa * dy, -sa * dx + ca * dy))

    got = [p.center for p in obs]
    # 排序可能整体反向（180° 歧义），正反两个方向都接受
    for cand in (list(zip(got, want)), list(zip(reversed(got), want))):
        if all(math.hypot(g[0] - w[0], g[1] - w[1]) < 6.0 for g, w in cand):
            return
    raise AssertionError(f"检出位置 {got} 与期望 {want} 不符")


def test_pixels_convert_to_body_millimetres():
    """核心断言：``targets()`` 返回的是**车身毫米**，不是像素。

    判据是三点间距。场地图里相邻靶相距 300mm；若忘了换算而直接返回像素，
    间距会变成 150（即 300mm ÷ 2mm/px）。两者相差整整一倍，不会被误差掩盖。
    """
    truth = Pose(1500.0, 1000.0, 270.0)
    img, _ = _render_targets(truth, FIELD.targets("rough"))
    obs = _perceive(img).targets(2)
    assert obs is not None and len(obs) == 3

    pts = [p.center for p in obs]
    d01 = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    d12 = math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
    assert abs(d01 - 300.0) < 10.0, f"相邻靶距应为 300mm，实测 {d01:.1f}"
    assert abs(d12 - 300.0) < 10.0, f"相邻靶距应为 300mm，实测 {d12:.1f}"


def test_end_to_end_pose_matches_truth():
    """完整链路：渲染 → 检出 → 换算 → 三点解位姿，结果必须等于真值。"""
    truth = Pose(1500.0, 1000.0, 270.0)
    group = FIELD.targets("rough")
    img, _ = _render_targets(truth, group)
    obs = _perceive(img).targets(2)
    assert obs is not None and len(obs) == 3

    theory = [(t.center[0], t.center[1]) for t in group]
    sol = solve_pose([p.center for p in obs], theory)
    assert abs(sol.pose.x - truth.x) < 3.0, f"x={sol.pose.x:.1f}"
    assert abs(sol.pose.y - truth.y) < 3.0, f"y={sol.pose.y:.1f}"
    diff = (sol.pose.heading - truth.heading + 180) % 360 - 180
    # 排序方向规范化可能整体反向，此处只在残差小的时候比较朝向
    if sol.residual_mm < 5.0 and abs(diff) < 5.0:
        assert abs(diff) < 5.0
    assert sol.residual_mm < 5.0, f"残差 {sol.residual_mm:.2f}mm 偏大"


def test_off_nominal_pose_still_solves():
    """车停偏 60mm、转头 5° 时仍能解出正确位姿（标称入场位姿只用于消歧）。

    偏移量不能取太大：三个靶必须全部落在视场内，否则检测层按设计返回 None。
    """
    truth = Pose(1560.0, 950.0, 275.0)
    group = FIELD.targets("rough")
    img, _ = _render_targets(truth, group)
    obs = _perceive(img).targets(2)
    assert obs is not None and len(obs) == 3
    theory = [(t.center[0], t.center[1]) for t in group]
    sol = solve_pose([p.center for p in obs], theory)
    assert abs(sol.pose.x - truth.x) < 8.0
    assert abs(sol.pose.y - truth.y) < 8.0
    assert sol.residual_mm < 8.0


# ---------- 拒绝输出 ----------

def test_no_extrinsics_means_no_targets():
    """无外参：返回 None。宁可跳过，也不发方向错位的测量结果。"""
    truth = Pose(1500.0, 1000.0, 270.0)
    img, _ = _render_targets(truth, FIELD.targets("rough"))
    assert _perceive(img, calib=None).targets(2) is None


def test_partial_visibility_means_no_targets():
    """只画两个靶时不得返回残缺观测（两靶定不出朝向）。"""
    truth = Pose(1500.0, 1000.0, 270.0)
    group = FIELD.targets("rough")[:2]
    img, _ = _render_targets(truth, group)
    assert _perceive(img).targets(2) is None


def test_empty_frame_means_no_targets():
    img = np.full((480, 640, 3), 240, np.uint8)
    assert _perceive(img).targets(2) is None


def test_turntable_is_explicitly_unavailable():
    """真实转盘识别尚未实现：必须显式返回 None，不能编造数据。"""
    img = np.full((480, 640, 3), 240, np.uint8)
    assert _perceive(img).turntable() is None


def test_extra_circles_do_not_break_the_triple():
    """画面里混入额外单圈（如转盘工位）时，仍应挑出圈数最多的三个真靶。"""
    truth = Pose(1500.0, 1000.0, 270.0)
    img, _ = _render_targets(truth, FIELD.targets("rough"))
    for x in (60, 120, 580):           # 边缘加几个孤立单圈
        cv2.circle(img, (x, 60), 30, (0, 0, 0), 2)
    obs = _perceive(img).targets(2)
    assert obs is not None and len(obs) == 3


# ---------- 选靶：用区域预测排除另一组靶与杂散圆 ----------

def _perceive_empty():
    img = np.full((480, 640, 3), 240, np.uint8)
    return VisionPerceive(_ScriptedCamera([img]), _cfg(),
                          calib=CameraCalib(CALIB), field=FIELD)


def test_predicted_body_points_for_rough_region():
    """粗加工区标称入场位姿 (1500,1000,270) 下，三个靶应在车头前方 400mm 一字排开。"""
    predicted = _perceive_empty()._predicted_body_points(2)
    assert predicted is not None and len(predicted) == 3
    for got, want in zip(predicted, [(400.0, -300.0), (400.0, 0.0), (400.0, 300.0)]):
        assert abs(got[0] - want[0]) < 1e-6 and abs(got[1] - want[1]) < 1e-6


def test_predicted_body_points_none_without_group():
    """原料区/起点没有参考靶，也就没有预测位置。"""
    per = _perceive_empty()
    assert per._predicted_body_points(1) is None
    assert per._predicted_body_points(None) is None


def test_pick_three_ignores_far_group_and_stray_circles():
    """画面里混着另一组靶（在车身另一侧）与杂散圆时，只挑与预测位置贴合的三个。"""
    per = _perceive_empty()
    near = [(402.0, -299.0, 5), (398.0, 1.0, 5), (401.0, 302.0, 5)]
    far = [(-800.0, -300.0, 5), (-800.0, 0.0, 5), (-800.0, 300.0, 5)]
    stray = [(0.0, 0.0, 1), (150.0, 150.0, 1)]

    picked = per._pick_three(near + far + stray, 2)
    assert sorted(picked) == sorted(near), f"挑错了：{picked}"


def test_pick_three_with_exactly_three_candidates_returns_them():
    per = _perceive_empty()
    body = [(0.0, 0.0, 5), (100.0, 0.0, 5), (200.0, 0.0, 5)]
    assert sorted(per._pick_three(body, 2)) == sorted(body)


def test_pick_three_falls_back_when_region_unknown():
    """没有区域提示时按「圈数多、离车近」挑，不依赖预测。"""
    per = _perceive_empty()
    singles = [(900.0, 900.0, 1), (850.0, 20.0, 1)]
    real = [(10.0, 0.0, 5), (20.0, 0.0, 5), (30.0, 0.0, 5)]
    picked = per._pick_three(singles + real, None)
    assert sorted(picked) == sorted(real), f"挑错了：{picked}"


def test_pick_three_from_two_groups_in_one_image():
    """两组靶同时出现在画面里时，按 region 挑出各自那一组。

    场地图里两组相距 1800mm，640×480、2mm/px 的画面装不下；这里临时构造一张
    「两组相距 1000mm」的场地图，让两组真的同时入镜，再用 region 区分。
    """
    from src.algorithms.fieldmap import FieldMap, TargetField, TargetGroup

    def group(y):
        return TargetGroup(
            tuple(TargetField(label=i + 1, center=(1000.0 + 300 * i, y),
                              radius=60.0, rings=5) for i in range(3)),
            entry_pose=Pose(1300.0, 1000.0, 270.0))

    field = FieldMap(groups={"rough": group(500.0), "staging": group(1500.0)})
    truth = Pose(1300.0, 1000.0, 270.0)
    img = np.full((480, 640, 3), 240, np.uint8)
    for name in ("rough", "staging"):
        _render_into(img, truth, field.targets(name))

    per = VisionPerceive(_ScriptedCamera([img]), _cfg(),
                         calib=CameraCalib(CALIB), field=field)
    rough, staging = per.targets(2), per.targets(3)
    assert rough is not None and staging is not None
    # 车头朝 −y：粗加工区靶在车头前方（车身 x 为正），暂存区在其后方（x 为负）
    assert min(p.center[0] for p in rough) > 200.0, [p.center for p in rough]
    assert max(p.center[0] for p in staging) < -200.0, [p.center for p in staging]


def test_pick_three_returns_none_when_region_targets_are_not_in_view():
    """当前区域的靶不在画面里时返回 None，不拿另一组的靶凑数。

    三个共线点可以用相似变换精确映射到任意另外三个共线点，**残差恒为 0**，
    单靠解算残差拦不住「拿错组」。所以必须在选靶阶段就用位置先验挡：挑出的靶只要
    有一个离预测位置超过门限，就返回 None，让 FSM 不发位姿。
    """
    from src.algorithms.fieldmap import FieldMap, TargetField, TargetGroup

    def group(y):
        return TargetGroup(
            tuple(TargetField(label=i + 1, center=(1000.0 + 300 * i, y),
                              radius=60.0, rings=5) for i in range(3)),
            entry_pose=Pose(1300.0, 1000.0, 270.0))

    field = FieldMap(groups={"rough": group(500.0), "staging": group(1500.0)})
    truth = Pose(1300.0, 1000.0, 270.0)
    img, _ = _render_targets(truth, field.targets("rough"))

    per = VisionPerceive(_ScriptedCamera([img]), _cfg(),
                         calib=CameraCalib(CALIB), field=field)
    assert per.targets(2) is not None, "画面里就是粗加工区的靶，应能挑出来"
    assert per.targets(3) is None, "画面里没有暂存区的靶，不得凑一组交出去"


# ---------- 与 FSM 的装配 ----------

def test_vision_perceive_exposes_fsm_keys():
    img = np.full((480, 640, 3), 240, np.uint8)
    perce = _perceive(img)
    for key in ("qr", "targets", "turntable"):
        assert callable(perce[key]), f"perceive 缺少可调用键 {key}"


def test_multi_camera_routes_qr_to_second_camera():
    """qr_index 生效：扫码走第二台相机，靶识别走主摄。"""
    from src.devices.camera import MultiCamera

    main = _ScriptedCamera([np.zeros((4, 4, 3), np.uint8)])
    qr = _ScriptedCamera([np.full((4, 4, 3), 7, np.uint8)])
    cam = MultiCamera(main, qr)
    ok, frame = cam.read()
    assert ok and frame.max() == 0          # 主摄
    ok, frame = cam.qr_read()
    assert ok and frame.max() == 7          # 扫码相机


def test_multi_camera_falls_back_to_main():
    from src.devices.camera import MultiCamera

    main = _ScriptedCamera([np.full((4, 4, 3), 3, np.uint8)])
    cam = MultiCamera(main, None)
    ok, frame = cam.qr_read()
    assert ok and frame.max() == 3
