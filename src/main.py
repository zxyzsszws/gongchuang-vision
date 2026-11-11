"""入口装配：python -m src.main --config config/default.yaml --mode sim|live|replay

v2.0.0 的分工是「视觉只测量、电控全规划」，因此装配比 v1.0.0 简单得多：
没有位姿滤波器、没有校准闭环、没有导航与动作指令——那些职责都已归还电控。

- sim：MockMcu（带里程漂移）+ SimScene 合成感知，虚拟时钟逐步驱动，完全可复现。
- live/replay：make_camera 相机帧驱动 VisionPerceive，链路为真实串口 SerialLink。
  live/replay 必须提供 ``camera_calib_file``（像素→车身毫米的外参），否则三靶
  观测无法换算成毫米，位姿解不出来；此时 ``targets()`` 返回 None，视觉会安静地
  不发位姿——这比拿像素当毫米发出去安全，但等于没有定位能力，故仍拒绝启动。
"""
import argparse
import time

from .algorithms.fieldmap import load_field_map
from .algorithms.types import Pose
from .config import cfg_get, load_config
from .core.fsm import MatchFSM
from .devices.link import DEFAULT_ROUTE, MockMcu, SerialLink
from .devices.sim_scene import SimScene
from .logsetup import setup_logging


class ExtrinsicsRequired(RuntimeError):
    """live/replay 缺少相机外参：像素无法换算成毫米，定位能力为零。"""


def _sim_waypoints(field, route):
    """把 ``route`` 里每个区域换成场地图上的标称停位，供 MockMcu 连线行驶。

    没有航点时 MockMcu 只能沿 +x 一直开，几段之后就跑到场外（x 上万毫米），
    模拟出来的「位姿」显然不可能是真实赛况。用 ``region_poses`` 连线后，车会
    在原料区、粗加工区、暂存区之间来回走，坐标始终落在场地范围内。
    """
    out = []
    for region in route:
        pose = field.pose_for_region(region)
        if pose is None:
            raise ValueError(
                f"场地图缺少区域 {region} 的标称停位：sim 无法生成行驶路线。"
                f"请在 field 配置的 region_poses 里补上 material/rough/staging")
        out.append(pose)
    return out


def _build_sim(cfg, field):
    sim_cfg = cfg.get("sim") or {}
    mcu = MockMcu(drift_mm_per_m=sim_cfg.get("drift_mm_per_m", 8.0),
                  drift_deg_per_m=sim_cfg.get("drift_deg_per_m", 0.1),
                  leg_mm=sim_cfg.get("leg_mm", 1200.0),
                  speed_mm_s=sim_cfg.get("speed_mm_s", 1200.0),
                  waypoints=_sim_waypoints(field, DEFAULT_ROUTE),
                  apply_pose=True)
    scene = SimScene(field, jitter_mm=sim_cfg.get("jitter_mm", 0.3),
                     omega01=sim_cfg.get("turntable_omega01", 600))
    return mcu, scene


def _build_live(cfg, field):
    from .algorithms.geometry import load_camera_calib
    from .devices.camera import make_camera
    from .devices.vision import VisionPerceive

    calib_path = cfg.get("camera_calib_file")
    if not calib_path:
        raise ExtrinsicsRequired(
            "live/replay 需要 camera_calib_file（相机安装外参）：像素观测必须换算到"
            "车身毫米才能解位姿。先跑 tools/calibrate_camera.py 标定，或在 config "
            "里指向已有标定文件。")
    calib = load_camera_calib(calib_path)
    cam = make_camera(cfg, cfg.get("mode", "live"))
    link = SerialLink(cfg["serial"]["port"], cfg["serial"]["baudrate"])
    return link, VisionPerceive(cam, cfg, calib=calib, field=field)


def run(cfg_path, mode, max_seconds=120.0):
    cfg = load_config(cfg_path)
    field = load_field_map(cfg_get(cfg, "field_file"))

    if mode == "sim":
        link, perceive = _build_sim(cfg, field)
    else:
        link, perceive = _build_live(dict(cfg, mode=mode), field)

    fsm = MatchFSM(cfg, link, field, perceive)

    if mode == "sim":
        # 虚拟时钟：固定 dt 逐步驱动，无墙钟竞态 → e2e 完全确定性、可复现。
        # 视觉是测量仪器，没有「跑完」这个终态，循环只由时限与安全停止终止。
        now, dt = 0.0, 0.02
        while now < max_seconds and fsm.state != "SAFESTOP":
            _drive_sim(link, perceive, now, dt)
            fsm.step(now)
            now += dt
    else:
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_seconds and fsm.state != "SAFESTOP":
            fsm.step(time.monotonic() - t0)
            time.sleep(0.02)
    return fsm


def _drive_sim(link, scene, now, dt):
    """推进 sim 世界：电控按路线行驶 → 场景按真值投影 → 转盘继续转。

    顺序有意义：先让电控走（可能刚好驶入新区域并发出 0x0005），再让场景按电控
    的**真值**位置生成观测，最后才驱动 FSM。这样 FSM 在 ``step`` 里读到的是同一
    时刻的世界状态，与真实系统的时序一致。
    """
    link.tick(now)
    scene.pose_truth = Pose(link.true_x, link.true_y, link.true_h)
    scene.advance(dt)


def main(argv=None):
    ap = argparse.ArgumentParser(description="gongchuang-vision 比赛入口")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--mode", choices=["sim", "live", "replay"], default=None,
                    help="缺省读配置 mode 键")
    ap.add_argument("--max-seconds", type=float, default=120.0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.get("log_dir", "logs"))  # 结构化 JSONL 运行日志（复盘用）
    mode = args.mode or cfg.get("mode", "sim")
    try:
        fsm = run(args.config, mode, max_seconds=args.max_seconds)
    except ExtrinsicsRequired as e:
        print(f"error: {e}")
        return 2
    for e in fsm.log_events:  # 状态迁移与关键事件
        print(e)
    # 视觉单元是测量仪器，不判断比赛何时结束，因此没有「完成」这个终态。
    # 退出码只区分「正常运行到时限」与「进入安全停止」。
    print(f"final_state={fsm.state} poses={fsm._pose_sent} "
          f"turntables={fsm._turntable_sent}")
    return 0 if fsm.state != "SAFESTOP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
