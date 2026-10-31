"""入口装配：python -m src.main --config config/default.yaml --mode sim|live|replay

run() 装配全栈（场地图 + 链路 + 滤波 + FSM + 感知）并按墙钟循环驱动 FSM.step
直至 DONE/SAFESTOP 或 max_seconds（spec §4 外层契约）。
- sim：MockMcu（带漂移推算）+ SimScene 合成感知，不触碰 cv2 相机。
- live/replay：make_camera 相机帧驱动 VisionPerceive（qr→WeChatQRCode，
  tri/rings→霍夫 + 外参换算到场地毫米），链路为真实串口 SerialLink。

坐标链路（Phase-4 A 项）：live/replay 的观测必须是**场地毫米**，否则校准会拿
像素当毫米、残差必然超限而全部降级。换算需要 ``camera_calib_file`` 外参；
缺该文件时 live/replay **拒绝启动**（宁停不错，spec §4 原则）。
"""
import argparse
import time

from .algorithms.fieldmap import load_field_map
from .config import cfg_get, load_config
from .core.fsm import MatchFSM
from .core.kalman import PoseFilter
from .devices.link import MockMcu, SerialLink
from .devices.sim_scene import SimScene
from .logsetup import setup_logging


class CalibrationRequired(RuntimeError):
    """live/replay 缺少相机外参：坐标系无法闭合，拒绝运行。"""


def _pose_hook(perceive, fsm):
    """把 FSM 最新的滤波位姿注入感知层（每拍在 step 前调用）。

    校准要用「上报位姿」把像素观测搬到场地坐标系；注入用**滤波后**位姿，
    与 FSM 内 calibrate 的输入同源，避免两者口径不一致。sim 模式不需要（SimScene
    直接读 MockMcu 位姿快照）。
    """
    if perceive is None:
        return lambda: None

    def before_step():
        perceive.pose = fsm._last

    return before_step


def run(cfg_path, mode, max_seconds=120.0):
    cfg = load_config(cfg_path)
    field = load_field_map(cfg_get(cfg, "field_file"))
    perceive = None
    if mode == "sim":
        # 漂移模型参数走 config（sim 节）；SimScene 持只读 link 引用，
        # 按当前位姿误差合成 field-as-seen（闭环校准）。
        sim_cfg = cfg.get("sim") or {}
        link = MockMcu(drift_mm_per_m=sim_cfg.get("drift_mm_per_m", 8.0),
                       drift_deg_per_m=sim_cfg.get("drift_deg_per_m", 0.1),
                       pose_hz=sim_cfg.get("pose_hz", 10.0))
        perceive = SimScene(field, link=link)
    else:
        from .algorithms.geometry import load_camera_calib
        from .devices.camera import make_camera
        from .devices.vision import VisionPerceive

        calib_path = cfg.get("camera_calib_file")
        if not calib_path:
            raise CalibrationRequired(
                "live/replay 需要 camera_calib_file（相机安装外参）：像素观测没有"
                "换算到场地毫米时校准无法成立。先跑 tools/calibrate_camera.py 标定，"
                "或在 config 里指向已有标定文件。")
        calib = load_camera_calib(calib_path)
        cam = make_camera(cfg, mode)
        link = SerialLink(cfg["serial"]["port"], cfg["serial"]["baudrate"])
        perceive = VisionPerceive(cam, cfg, calib=calib, field=field)

    fsm = MatchFSM(cfg, link, PoseFilter(), field, perceive)
    hook = _pose_hook(perceive if mode != "sim" else None, fsm)

    if mode == "sim":
        # 虚拟时钟：固定 dt 逐步驱动，无墙钟竞态 → e2e 完全确定性、可复现。
        now, dt = 0.0, 0.02
        while now < max_seconds and fsm.state not in ("DONE", "SAFESTOP"):
            hook()
            fsm.step(now)
            now += dt
    else:
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_seconds and fsm.state not in ("DONE", "SAFESTOP"):
            hook()
            fsm.step(time.monotonic() - t0)
            time.sleep(0.02)
    return fsm


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
    except CalibrationRequired as e:
        print(f"error: {e}")
        return 2
    for e in fsm.log_events:  # 状态迁移与关键事件（结构化日志待 Phase-4 B 项接入）
        print(e)
    print(f"final_state={fsm.state}")
    return 0 if fsm.state == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
