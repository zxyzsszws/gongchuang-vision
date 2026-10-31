"""相机源：USB 摄像头 / 视频文件 / 图片序列目录 + 工厂。

CameraSource 协议：read() -> (ok, frame|None)，与 cv2.VideoCapture.read() 同形，
上层无需关心具体来源。

双相机分工（与方案文档一致）：主摄（``camera.main_index``）负责环识别，
扫码相机（``camera.qr_index``，花雀广角）专职扫码；``qr_index`` 缺省时扫码
复用主摄（此前该配置项一直是死键，现在生效）。

``MultiCamera`` 把主摄的 ``read()`` 与扫码相机的 ``qr_read()`` 合成一个对象，
VisionPerceive 因此无需区分单/双相机。
"""
import glob
import os
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class CameraSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...


class USBCamera:
    def __init__(self, index, width=640, height=480, fps=30):
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()


class VideoFile:
    def __init__(self, path):
        self._cap = cv2.VideoCapture(path)

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()


class ImageFolder:
    def __init__(self, dir_path, loop=False):
        self._files = sorted(glob.glob(os.path.join(dir_path, "*.png")))
        self._i, self._loop = 0, loop

    def read(self):
        if self._i >= len(self._files):
            if not self._loop:
                return False, None
            self._i = 0
        frame = cv2.imread(self._files[self._i])
        self._i += 1
        return frame is not None, frame


class MultiCamera:
    """主摄 + 扫码相机：read() 走主摄，qr_read() 走扫码相机（缺省回落到主摄）。"""

    def __init__(self, main, qr=None):
        self._main, self._qr = main, qr

    def read(self):
        return self._main.read()

    def qr_read(self):
        if self._qr is None:
            return self._main.read()
        ok, frame = self._qr.read()
        return (ok, frame) if ok and frame is not None else self._main.read()

    def release(self):
        self._main.release()
        if self._qr is not None:
            self._qr.release()


def make_camera(cfg, mode):
    if mode == "live":
        cam_cfg = cfg["camera"]
        main = USBCamera(cam_cfg["main_index"], cam_cfg["width"], cam_cfg["height"])
        qr_index = cam_cfg.get("qr_index")
        if qr_index is None or qr_index == cam_cfg["main_index"]:
            return main
        return MultiCamera(main, USBCamera(qr_index, cam_cfg["width"], cam_cfg["height"]))
    if mode == "replay":
        path = cfg["replay_path"]
        return VideoFile(path) if path.endswith((".mp4", ".avi")) else ImageFolder(path)
    return ImageFolder(cfg["sim_images"], loop=True)
