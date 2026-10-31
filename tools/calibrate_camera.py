"""相机外参标定工具：由「图像像素点 ↔ 车体毫米点」成对实测数据拟合外参。

不需要棋盘格，不需要相机内参，量尺子即可。

用法（推荐两点法，每次用 ``--pair`` 给一对点）：

    python tools/calibrate_camera.py \
        --pair 170,240,-150,0 \
        --pair 470,240,150,0 \
        --out config/camera_calib.yaml

``--pair`` 的格式是 ``px_x,px_y,mm_x,mm_y``。**必须用 ``--pair`` 而不是分开的
--px/--mm**：车体毫米坐标常为负值，分开传会被 argparse 当成命令选项而报错。

多点法（更准；≥4 点可加 ``--homography`` 拟合透视单应，三色环不在同一景深时首选）：

    python tools/calibrate_camera.py --csv calib_points.csv --out config/camera_calib.yaml --homography

CSV 每行 ``px_x,px_y,mm_x,mm_y``，``#`` 开头为注释。

坐标系提醒（务必对齐）：
- **车体坐标系**以相机光轴正下方地面点为原点、车头方向 +y、右侧 +x，单位 mm ——
  与校准算法「pose = 车体原点在场地中的位姿」的口径一致。
- 图像 y 轴朝下，故 rot_deg 通常带 ±90°/180° 量级。工具会打印每点残差，
  残差 > 10mm 说明点配对错了、单位不是 mm，或标定点不在同一平面。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.geometry import fit_homography, fit_similarity  # noqa: E402


def _floats(text, n, where):
    parts = text.replace("\t", ",").split(",")
    if len(parts) != n:
        raise SystemExit(f"{where}: need {n} comma-separated numbers, got {text!r}")
    try:
        return tuple(float(v) for v in parts)
    except ValueError:
        raise SystemExit(f"{where}: non-numeric value in {text!r}")


def _pairs(args):
    if args.csv:
        src, dst = [], []
        with open(args.csv, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                px_x, px_y, mm_x, mm_y = _floats(line, 4, f"{args.csv}:{n}")
                src.append((px_x, px_y))
                dst.append((mm_x, mm_y))
        if len(src) < 2:
            raise SystemExit(f"{args.csv}: need >=2 point rows")
        return src, dst
    if not args.pair:
        raise SystemExit("need at least two --pair px_x,px_y,mm_x,mm_y")
    if len(args.pair) < 2:
        raise SystemExit("need at least two --pair entries")
    src, dst = [], []
    for i, text in enumerate(args.pair, 1):
        px_x, px_y, mm_x, mm_y = _floats(text, 4, f"--pair #{i}")
        src.append((px_x, px_y))
        dst.append((mm_x, mm_y))
    return src, dst


def _residuals(model, src, dst):
    out = []
    for s, d in zip(src, dst):
        got = model.apply(s)
        out.append(((got[0] - d[0]) ** 2 + (got[1] - d[1]) ** 2) ** 0.5)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="相机像素 → 车体毫米 外参标定")
    ap.add_argument("--pair", action="append",
                    help="一对实测点 px_x,px_y,mm_x,mm_y（至少两对）")
    ap.add_argument("--csv", help="多点模式：每行 px_x,px_y,mm_x,mm_y")
    ap.add_argument("--homography", action="store_true",
                    help="拟合 3x3 单应（需 >=4 点；透视变形明显时用）")
    ap.add_argument("--auto-scale-from-tri", action="store_true",
                    help="尺度改由三色环半径实时自估（远近都要看到三色环时用）")
    ap.add_argument("--out", required=True, help="输出 yaml 路径")
    args = ap.parse_args(argv)

    src, dst = _pairs(args)
    if args.homography:
        if len(src) < 4:
            raise SystemExit("--homography needs >=4 points")
        model = fit_homography(src, dst)
        lines = ["# 相机外参（像素 → 车体毫米，单应）：tools/calibrate_camera.py 生成",
                 "homography:"]
        lines += [f"  - [{h[0]:.10g}, {h[1]:.10g}, {h[2]:.10g}]" for h in model.matrix]
    else:
        model = fit_similarity(src, dst)
        lines = ["# 相机外参（像素 → 车体毫米，相似变换）：tools/calibrate_camera.py 生成",
                 "image_to_robot:",
                 f"  scale_mm_per_px: {model.scale:.10g}",
                 f"  rot_deg: {model.rot_deg:.10g}",
                 f"  tx_mm: {model.tx:.10g}",
                 f"  ty_mm: {model.ty:.10g}"]
    if args.auto_scale_from_tri:
        lines.append("auto_scale_from_tri: true")
    lines.append("")

    res = _residuals(model, src, dst)
    worst = max(res)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}")
    for i, r in enumerate(res, 1):
        print(f"  point {i}: residual {r:.2f} mm")
    print(f"max residual {worst:.2f} mm")
    if worst > 10.0:
        print("warning: residual > 10mm —— 检查点配对、单位(mm)、标定点是否共面",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
