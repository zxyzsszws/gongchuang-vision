"""相机源三实现：USB 摄像头 / 视频文件 / 图片序列目录 + 工厂。

CameraSource 协议：read() -> (ok, frame|None)，与 cv2.VideoCapture.read() 同形，
上层（Task 12 管道）无需关心具体来源。
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


def make_camera(cfg, mode):
    if mode == "live":
        return USBCamera(cfg["camera"]["main_index"], cfg["camera"]["width"], cfg["camera"]["height"])
    if mode == "replay":
        path = cfg["replay_path"]
        return VideoFile(path) if path.endswith((".mp4", ".avi")) else ImageFolder(path)
    return ImageFolder(cfg["sim_images"], loop=True)
