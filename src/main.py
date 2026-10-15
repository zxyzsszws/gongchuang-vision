"""入口装配：python -m src.main --config config/default.yaml --mode sim|live|replay

run() 装配全栈（场地图 + 链路 + 滤波 + FSM + 感知）并按墙钟循环驱动 FSM.step
直至 DONE/SAFESTOP 或 max_seconds（spec §4 外层契约）。
- sim：MockMcu（带漂移推算）+ SimScene 合成感知，不触碰 cv2 相机。
- live/replay：make_camera 相机帧驱动 perceive（qr→decode_qr，rings/tri→
  detect_rings/detect_tri_ring），链路为真实串口 SerialLink。
"""
import argparse
import time

from .algorithms.fieldmap import load_field_map
from .config import cfg_get, load_config
from .core.fsm import MatchFSM
from .core.kalman import PoseFilter
from .devices.link import MockMcu, SerialLink
from .devices.sim_scene import SimScene


def _make_vision_perceive(cam, cfg):
    # 仅 live/replay 分支导入：sim 路径不加载相机/解码器依赖。
    from .algorithms.qr_decode import decode_qr
    from .algorithms.ring_detect import detect_rings, detect_tri_ring

    ring_cfg = cfg["ring"]
    geo, hough = ring_cfg["geo"], ring_cfg.get("hough")
    # 分组参数配置驱动，缺省与 Task 8 定标值一致（120-200mm，共线容差 5°）。
    spacing = (ring_cfg.get("spacing_min_mm", 120), ring_cfg.get("spacing_max_mm", 200))
    collinear_tol = ring_cfg.get("collinear_tol_deg", 5)
    cache = {"frame": None, "rings": None}

    def read():
        # 缓存最近帧：FSM 每拍至多调用一个感知键 → 等效「每轮读帧一次」；
        # 新帧到达才失效圆环缓存，tri 与 rings 共享同一次霍夫检出。
        ok, img = cam.read()
        if ok and img is not None:
            cache["frame"], cache["rings"] = img, None
        return cache["frame"]

    def rings_of_frame():
        img = read()
        if img is None:
            return []
        if cache["rings"] is None:
            cache["rings"] = detect_rings(img, geo, hough)
        return cache["rings"]

    def qr():
        img = read()
        return decode_qr(img) if img is not None else None

    def tri():
        return detect_tri_ring(rings_of_frame(), spacing, collinear_tol)

    def rings():
        return rings_of_frame()

    return {"qr": qr, "tri": tri, "rings": rings}


def run(cfg_path, mode, max_seconds=120.0):
    cfg = load_config(cfg_path)
    field = load_field_map(cfg_get(cfg, "field_file"))
    if mode == "sim":
        link = MockMcu(drift_mm_per_m=20.0, drift_deg_per_m=0.1, pose_hz=10.0)
        perceive = SimScene(field)
    else:
        from .devices.camera import make_camera
        cam = make_camera(cfg, mode)
        link = SerialLink(cfg["serial"]["port"], cfg["serial"]["baudrate"])
        # live/replay: perceive 由相机帧驱动（每帧先 cam.read 再跑检测器）
        perceive = _make_vision_perceive(cam, cfg)
    fsm = MatchFSM(cfg, link, PoseFilter(), field, perceive)
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_seconds and fsm.state not in ("DONE", "SAFESTOP"):
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
    mode = args.mode or load_config(args.config).get("mode", "sim")
    fsm = run(args.config, mode, max_seconds=args.max_seconds)
    for e in fsm.log_events:  # 状态迁移与关键事件日志（Task 13 将替换为结构化日志）
        print(e)
    print(f"final_state={fsm.state}")
    return 0 if fsm.state == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
