"""标定工具 CLI 回归：负毫米坐标必须可用、输出可被 load_camera_calib 读回。

负值是关键坑：``--px 170,240 --mm -150,0`` 会被 argparse 当成命令选项而报错，
故工具改用单一 ``--pair px_x,px_y,mm_x,mm_y`` 形式。下面的测试固定住这一契约。
"""
import math

import pytest

from src.algorithms.geometry import Similarity, load_camera_calib
from tools.calibrate_camera import main


def test_two_pairs_recover_transform_including_negative_mm(tmp_path):
    out = tmp_path / "calib.yaml"
    rc = main(["--pair", "170,240,-150,0",
               "--pair", "470,240,150,0",
               "--out", str(out)])
    assert rc == 0
    calib = load_camera_calib(str(out))
    tr = calib.image_to_robot
    assert isinstance(tr, Similarity)
    # 1mm/px、无旋转、图像中心 (320,240) → 车体原点
    assert abs(tr.scale - 1.0) < 1e-9
    assert abs(tr.rot_deg) < 1e-9
    x, y = tr.apply((320, 240))
    assert abs(x) < 1e-9 and abs(y) < 1e-9
    # 车体坐标可为负：图像左侧点 → 车体负 x
    x, y = tr.apply((170, 240))
    assert abs(x + 150.0) < 1e-9


def test_rotated_transform(tmp_path):
    # 图像 y 朝下、车头朝上 → 典型 180° 安装；用两点反推
    truth = Similarity(scale=1.5, rot_deg=180.0, tx=480.0, ty=360.0)
    p1, p2 = (100.0, 120.0), (500.0, 400.0)
    d1, d2 = truth.apply(p1), truth.apply(p2)
    out = tmp_path / "calib.yaml"
    rc = main(["--pair", f"{p1[0]},{p1[1]},{d1[0]},{d1[1]}",
               "--pair", f"{p2[0]},{p2[1]},{d2[0]},{d2[1]}",
               "--out", str(out)])
    assert rc == 0
    got = load_camera_calib(str(out)).image_to_robot
    assert abs(got.scale - truth.scale) < 1e-6
    # 角度相差 180° 等价（相似变换旋转符号约定）
    diff = abs(((got.rot_deg - truth.rot_deg + 180) % 360) - 180)
    assert diff < 1e-4
    for pt in (p1, p2):
        want, have = truth.apply(pt), got.apply(pt)
        assert math.hypot(want[0] - have[0], want[1] - have[1]) < 1e-6


def test_csv_mode_and_homography(tmp_path):
    csv = tmp_path / "pts.csv"
    csv.write_text("# px_x,px_y,mm_x,mm_y\n"
                   "0,0,0,0\n"
                   "640,0,640,0\n"
                   "640,480,640,480\n"
                   "0,480,0,480\n", encoding="utf-8")
    out = tmp_path / "calib.yaml"
    assert main(["--csv", str(csv), "--homography", "--out", str(out)]) == 0
    calib = load_camera_calib(str(out))
    x, y = calib.image_to_robot.apply((320, 240))
    assert abs(x - 320) < 1e-6 and abs(y - 240) < 1e-6


def test_rejects_too_few_points(tmp_path):
    with pytest.raises(SystemExit):
        main(["--pair", "0,0,0,0", "--out", str(tmp_path / "x.yaml")])
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "x.yaml")])


def test_homography_requires_four_points(tmp_path):
    with pytest.raises(SystemExit):
        main(["--pair", "0,0,0,0", "--pair", "1,0,1,0",
              "--pair", "1,1,1,1", "--homography", "--out", str(tmp_path / "x.yaml")])


def test_warns_on_large_residual(tmp_path, capsys):
    # 三点共线但给了一组不自洽的点 → 残差大 → 非零退出并告警
    rc = main(["--pair", "0,0,0,0",
               "--pair", "100,0,100,0",
               "--pair", "0,100,0,50",   # 与相似变换不自洽
               "--out", str(tmp_path / "x.yaml")])
    err = capsys.readouterr().err
    assert rc == 1 and "residual" in err
