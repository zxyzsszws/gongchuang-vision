"""入口装配：python -m src.main --config config/default.yaml --mode sim|live|replay

V2.2.0 的视觉只做三件事，全部**按需应答**，所以装配很轻：没有位姿滤波器、
没有校准闭环、没有周期上报，也没有场地图——视觉不再需要知道自己在场地哪里。

- sim：MockMcu + SimScene 合成感知，虚拟时钟逐步驱动，完全可复现。
- live/replay：make_camera 相机帧驱动 VisionPerceive，链路为真实串口 SerialLink。
  ``camera_calib_file`` 应当提供：任务 2（转盘颜色）的尺度只能来自外参，缺了会
  一律回「没准备好」；任务 3（对准主圆）的尺度从主圆外圈实测反算，不依赖外参，
  只有外圈也量不到时才回「没准备好」。两种情况都不会瞎猜尺度。
"""
import argparse
import time

from .config import load_config
from .core.fsm import MatchFSM
from .core.protocol import TASK_ALIGN, TASK_SCAN, TASK_TURNTABLE
from .devices.link import MockMcu, SerialLink
from .devices.sim_scene import SimScene
from .logsetup import setup_logging


def _build_sim(cfg):
    sim_cfg = cfg.get("sim") or {}
    # 视觉不再上报位姿，模拟器也就不需要里程漂移这些参数了。
    mcu = MockMcu()
    scene = SimScene(jitter_mm=sim_cfg.get("jitter_mm", 0.3),
                     omega01=sim_cfg.get("turntable_omega01", 600),
                     stop_after_s=sim_cfg.get("turntable_stop_after_s"))
    return mcu, scene


def _build_live(cfg, mode):
    from .algorithms.geometry import load_camera_calib
    from .devices.camera import make_camera
    from .devices.vision import VisionPerceive

    calib = None
    calib_path = cfg.get("camera_calib_file")
    if calib_path:
        calib = load_camera_calib(calib_path)
    if calib is None:
        # 启动时不拦，但要大声说出来：没有外参时任务 2 会一直回「没准备好」，
        # 现场很容易误判成「没看到东西」而去乱调阈值。任务 3 不受影响——
        # 它的尺度从主圆外圈实测反算，只有外圈也量不到时才用外参兜底。
        print("[警告] 没有可用相机外参：任务 2（转盘颜色）将一律回 ST_NOT_READY。"
              "任务 3（对准主圆）仍可工作，但外圈量不到时也会失败。"
              "请按 README §6 标定并设置 camera_calib_file。")
    cam = make_camera(cfg, mode)
    link = SerialLink(cfg["serial"]["port"], cfg["serial"]["baudrate"])
    return link, VisionPerceive(cam, cfg, calib=calib)


def _sim_dialog(mcu, now):
    """模拟电控按比赛流程问话：扫码 → 问取料点颜色 → 问对准。

    这里刻意写得像真实电控：**只在需要的时候问，问完等结果**。视觉不知道
    「现在该干什么」，完全由这几帧请求驱动。
    """
    if TASK_SCAN not in mcu.requests:
        mcu.request_scan(now)
    elif mcu.plan is not None and TASK_TURNTABLE not in mcu.requests:
        mcu.request_turntable(now)
    elif mcu.colors and TASK_ALIGN not in mcu.requests:
        mcu.request_align(now)


def run(cfg_path, mode, max_seconds=120.0):
    cfg = load_config(cfg_path)

    if mode == "sim":
        link, perceive = _build_sim(cfg)
    else:
        link, perceive = _build_live(dict(cfg, mode=mode), mode)

    fsm = MatchFSM(cfg, link, perceive)

    if mode == "sim":
        # 虚拟时钟：固定 dt 逐步驱动，无墙钟竞态 → e2e 完全确定性、可复现。
        now, dt = 0.0, 0.02
        while now < max_seconds and fsm.state != "SAFESTOP":
            link.tick(now)
            _sim_dialog(link, now)
            perceive.advance(dt)
            fsm.step(now)
            now += dt
    else:
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_seconds and fsm.state != "SAFESTOP":
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
    fsm = run(args.config, mode, max_seconds=args.max_seconds)
    for e in fsm.log_events:  # 状态迁移与关键事件
        print(e)
    # 视觉单元是按需应答的仪器，不判断比赛何时结束，因此没有「完成」这个终态。
    # 退出码只区分「正常运行到时限」与「进入安全停止」。
    print(f"final_state={fsm.state} "
          f"counts={fsm.counts} bad={fsm.bad_requests} resends={fsm.resends}")
    return 0 if fsm.state != "SAFESTOP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
