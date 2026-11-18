"""把「合成图测试」做成看得见的图：渲染一帧 → 真检测 → 叠加真值与检出。

为什么需要这个工具：``tests/test_live_pipeline.py`` 里的合成图只断言「检出的
靶心与真值差小于 6mm 就算通过」，**通过时什么都不输出**。看不到图，就没法判断
阈值是「刚好压线通过」还是「有富余」，也没法看检测框、误检、靶被裁掉这些
只存在于画面里的问题。

这个脚本走的是与测试**完全相同**的一条路径：

    场地图靶心 --真位姿的逆--> 车身毫米 --外参的逆--> 像素 --cv2.circle-->
    图像 --detect_targets--> 像素观测 --外参--> 车身毫米 --solve_pose--> 位姿

所以图上看到的绿圈就是检测真正找到的圆，红叉是真值该在的位置，两者不重合
就是误差。参数从 ``config/default.yaml`` 读，与 ``--mode live`` 一致。

用法（在本仓库根目录执行）：

    .venv\\Scripts\\python.exe tools/viz_synth.py
    .venv\\Scripts\\python.exe tools/viz_synth.py --region rough --heading 275
    .venv\\Scripts\\python.exe tools/viz_synth.py --sweep

``--sweep`` 会在同一张图上生成多组「位姿偏移 / 噪声 / 靶半径」的对比，用来判断
判据还有多少余量。输出默认写到 ``logs/viz/``。
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.fieldmap import load_field_map           # noqa: E402
from src.algorithms.geometry import CameraCalib, Similarity   # noqa: E402
from src.algorithms.localize import resolve_order            # noqa: E402
from src.algorithms.ring_detect import detect_targets        # noqa: E402
from src.algorithms.types import Pose                        # noqa: E402
from src.core import protocol                                # noqa: E402
from src.devices.vision import VisionPerceive                # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 tests/test_live_pipeline.py 保持一致的默认外参：2mm/px、无旋转
SCALE = 2.0
CALIB = Similarity(scale=SCALE, rot_deg=0.0,
                   tx=-320.0 * SCALE, ty=-240.0 * SCALE)
INV = CALIB.invert()

W, H = 640, 480


# ------------------------------------------------------------------ 渲染
def body_point(pose, field_pt):
    """场地坐标 → 车身坐标（真位姿的逆变换）。"""
    a = math.radians(pose.heading)
    ca, sa = math.cos(a), math.sin(a)
    dx = field_pt[0] - pose.x
    dy = field_pt[1] - pose.y
    return (ca * dx + sa * dy, -sa * dx + ca * dy)


def render(pose, group, rings_mm=(40, 60, 80), thickness=2,
           noise=0.0, blur=0, radius_mm=60):
    """按真位姿渲染一帧。返回 (img, 真值像素圆心列表)。"""
    img = np.full((H, W, 3), 240, np.uint8)
    px_per_mm = 1.0 / SCALE
    truths = []

    for t in group:
        bx, by = body_point(pose, t.center)
        px = INV.apply((bx, by))
        truths.append(px)
        for r_mm in rings_mm:
            r_px = int(round(r_mm * px_per_mm))
            if r_px < 1:
                continue
            cv2.circle(img, (int(round(px[0])), int(round(px[1]))),
                       r_px, (0, 0, 0), thickness)

    # 靶心数字：真实场景里印在中心，合成图里画一个小圆点占位
    for px in truths:
        cv2.circle(img, (int(round(px[0])), int(round(px[1]))), 3, (0, 0, 0), -1)

    if blur:
        k = blur * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    if noise:
        n = np.random.default_rng(7).normal(0, noise, img.shape)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return img, truths


# ------------------------------------------------------------------ 检测
def load_cfg():
    with open(os.path.join(REPO, "config", "default.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # 合成图里靶的像素半径是 20/30/40（rings/SCALE），搜索区间必须盖住
    cfg["target"] = dict(cfg["target"], search_radius=[12, 60], min_rings=3)
    return cfg


class _OneShotCamera:
    """把单帧包成 CameraSource，让 VisionPerceive 走与 live 相同的代码路径。"""

    def __init__(self, img):
        self._img = img
        self._used = False

    def read(self):
        if self._used:
            return False, None
        self._used = True
        return True, self._img


def analyze(img, calib=CALIB, cfg=None):
    """真实链路：像素检测 → 车身毫米 → 解位姿。"""
    cfg = cfg or load_cfg()
    field = load_field_map(os.path.join(REPO, "config", "field.example.yaml"))

    px_obs = detect_targets(img, cfg["target"])
    body = [(calib.apply(t.center), t) for t in px_obs]

    vp = VisionPerceive(_OneShotCamera(img), cfg,
                        calib=CameraCalib(calib), field=field)
    return {"px": px_obs, "body": body, "perceive": vp, "cfg": cfg,
            "field": field}


# ------------------------------------------------------------------ 画图
COLOR_TRUE = (60, 60, 220)     # 真值：红（BGR）
COLOR_DET = (90, 190, 90)      # 检出：绿
COLOR_RING = (200, 140, 60)    # 单个圆：蓝
COLOR_TEXT = (40, 40, 40)


def draw(img, res, truth_px, pose=None, title=""):
    """把检测结果叠加到图上，返回新图（不改原图）。"""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

    # 检出的单个圆（用于看「为什么凑不成靶」）
    for t in res["px"]:
        cx, cy = int(round(t.center[0])), int(round(t.center[1]))
        cv2.drawMarker(out, (cx, cy), COLOR_RING, cv2.MARKER_CROSS, 14, 1)

    # 检出并聚成的靶
    for t in res["px"]:
        cx, cy = int(round(t.center[0])), int(round(t.center[1]))
        r = int(round(t.radius_px))
        cv2.circle(out, (cx, cy), r, COLOR_DET, 2)
        cv2.putText(out, f"{t.ring_count}ring", (cx - 20, cy - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_DET, 1, cv2.LINE_AA)

    # 真值靶心；误差按**最近**检出配对，不能按下标——
    # 聚类后的顺序与真值顺序无关，按下标配会得出几百像素的假误差。
    for p in truth_px:
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.drawMarker(out, (x, y), COLOR_TRUE, cv2.MARKER_TILTED_CROSS, 18, 2)
        if res["px"]:
            near = min(res["px"],
                       key=lambda t: math.hypot(t.center[0] - p[0],
                                                t.center[1] - p[1]))
            err = math.hypot(near.center[0] - p[0], near.center[1] - p[1])
            # 标签放到外圈之外，避免压在环上
            dy = int(round(near.radius_px)) + 12
            cv2.putText(out, f"{err:.1f}px", (x - 18, y + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TRUE, 1, cv2.LINE_AA)

    # 角标说明
    lines = [title] if title else []
    lines.append(f"检出 {len(res['px'])} 个靶（要求 3）")
    if pose is not None:
        lines.append(f"真值 ({pose.x:.0f}, {pose.y:.0f}) "
                     f"{pose.heading:.1f}deg")
    y0 = 18
    for s in lines:
        cv2.putText(out, s, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COLOR_TEXT, 1, cv2.LINE_AA)
        y0 += 17
    return out


def solve_and_report(res, pose_true, region):
    """跑完整解位姿（含 180° 消歧），返回 (解, 位置误差 mm, 朝向误差 deg)。

    ``targets()`` 返回的已经是车身毫米，可直接交给 ``resolve_order``；
    theory 用场地图里该组三靶的**编号顺序**坐标。
    """
    obs = res["perceive"].targets(region)
    if obs is None:
        return None, None, None

    group = res["field"].group_for_region(region)
    theory = [(t.center[0], t.center[1]) for t in group]
    body = [p.center for p in obs]
    prior = res["field"].entry_pose_for_region(region)

    try:
        cand = resolve_order(body, theory, prior=prior)
    except Exception:
        return None, None, None

    sol = cand.solution.pose
    pos_err = math.hypot(sol.x - pose_true.x, sol.y - pose_true.y)
    head_err = abs((sol.heading - pose_true.heading + 180) % 360 - 180)
    return sol, pos_err, head_err


# ------------------------------------------------------------------ 主流程
def scene_one(args):
    field = load_field_map(os.path.join(REPO, "config", "field.example.yaml"))
    group = field.targets(args.region)
    pose = Pose(args.x, args.y, args.heading)

    img, truths = render(pose, group, noise=args.noise, blur=args.blur)
    res = analyze(img)

    title = f"{args.region}  noise={args.noise}  blur={args.blur}"
    out = draw(img, res, truths, pose, title)
    return out


def scene_sweep(args):
    """一组对比：位姿偏移 / 噪声 / 模糊，各自单独变化。"""
    field = load_field_map(os.path.join(REPO, "config", "field.example.yaml"))
    group = field.targets(args.region)

    cases = [
        ("基准", Pose(1500, 1000, 270), 0.0, 0),
        ("横移 80mm", Pose(1580, 1000, 270), 0.0, 0),
        ("朝向偏 5deg", Pose(1500, 1000, 275), 0.0, 0),
        ("高斯噪声 sd=18", Pose(1500, 1000, 270), 18.0, 0),
        ("模糊 k=5", Pose(1500, 1000, 270), 0.0, 2),
        ("噪声+模糊", Pose(1500, 1000, 270), 18.0, 2),
    ]

    tiles = []
    for name, pose, noise, blur in cases:
        img, truths = render(pose, group, noise=noise, blur=blur)
        res = analyze(img)
        n = len(res["px"])
        tiles.append(draw(img, res, truths, pose,
                          f"{name} -> {n} 个靶"))

    rows = []
    for i in range(0, len(tiles), 3):
        row = tiles[i:i + 3]
        while len(row) < 3:
            row.append(np.full_like(tiles[0], 255))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="合成图可视化")
    ap.add_argument("--region", default="rough", choices=["rough", "staging"])
    # 默认位姿取该区域的标称入场位姿，否则另一组靶会跑到画面外
    ap.add_argument("--x", type=float, default=None)
    ap.add_argument("--y", type=float, default=None)
    ap.add_argument("--heading", type=float, default=None)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="高斯噪声标准差（灰度级）")
    ap.add_argument("--blur", type=int, default=0,
                    help="高斯模糊核半径（0=不模糊）")
    ap.add_argument("--sweep", action="store_true",
                    help="生成 6 种扰动的对比图")
    ap.add_argument("--out", default=os.path.join(REPO, "logs", "viz"))
    args = ap.parse_args(argv)

    # 缺省位姿从场地图的 entry_pose 取，保证该组三靶一定在视野内
    field0 = load_field_map(os.path.join(REPO, "config", "field.example.yaml"))
    region_code = (protocol.REGION_ROUGH if args.region == "rough"
                   else protocol.REGION_STAGING)
    base = field0.entry_pose_for_region(region_code) or Pose(0, 0, 0)
    args.x = base.x if args.x is None else args.x
    args.y = base.y if args.y is None else args.y
    args.heading = base.heading if args.heading is None else args.heading

    os.makedirs(args.out, exist_ok=True)

    if args.sweep:
        img = scene_sweep(args)
        name = f"sweep-{args.region}.png"
    else:
        img = scene_one(args)
        name = (f"{args.region}-x{int(args.x)}-y{int(args.y)}"
                f"-h{int(args.heading)}-n{int(args.noise)}-b{args.blur}.png")

    path = os.path.join(args.out, name)
    cv2.imwrite(path, img)
    print(f"wrote {path}")

    # 顺便把数值也打出来，便于与图对照
    if not args.sweep:
        field = load_field_map(os.path.join(REPO, "config", "field.example.yaml"))
        pose = Pose(args.x, args.y, args.heading)
        img2, truths = render(pose, field.targets(args.region),
                             noise=args.noise, blur=args.blur)
        res = analyze(img2)
        print(f"检出靶数: {len(res['px'])}")
        for p in truths:
            if not res["px"]:
                break
            near = min(res["px"],
                       key=lambda t: math.hypot(t.center[0] - p[0],
                                                t.center[1] - p[1]))
            err = math.hypot(near.center[0] - p[0], near.center[1] - p[1])
            print(f"  真值靶心 ({p[0]:.0f},{p[1]:.0f}): "
                  f"最近检出误差 {err:.2f}px ({err * SCALE:.2f}mm), "
                  f"圈数 {near.ring_count}, 半径 {near.radius_px:.1f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
