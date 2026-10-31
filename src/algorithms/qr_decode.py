"""WeChatQRCode 解码封装（algorithms 层，仅 live/replay 使用）。

模型优先级：
1. config ``camera.qr_model_dir`` 指向的目录（含 detect.prototxt / detect.caffemodel /
   sr.prototxt / sr.caffemodel）—— 与方案文档一致，超分模型显著提升远距离/小码识读率；
2. 目录缺失或不完整时回退 ``WeChatQRCode()`` 无参构造（OpenCV 自带缺省模型），
   功能可用但辨识率低于加载模型——现场务必补上模型文件。

检测器惰性单例：首次 decode_qr() 才构建（构建较重，避免 sim/测试路径付初始化成本）。
"""
import os

import cv2

_MODEL_FILES = ("detect.prototxt", "detect.caffemodel",
                "sr.prototxt", "sr.caffemodel")
_DET = None
_DET_SOURCE = None


def _model_paths(model_dir):
    """返回 4 个模型文件的绝对路径；不全则返回 None。"""
    if not model_dir:
        return None
    paths = [os.path.join(model_dir, name) for name in _MODEL_FILES]
    return paths if all(os.path.isfile(p) for p in paths) else None


def build_detector(model_dir=None):
    paths = _model_paths(model_dir)
    if paths:
        return cv2.wechat_qrcode_WeChatQRCode(*paths), "files"
    return cv2.wechat_qrcode_WeChatQRCode(), "default"


def configure(model_dir=None):
    """显式指定模型目录并重建检测器（main 装配时调用；返回模型来源供日志）。"""
    global _DET, _DET_SOURCE
    _DET, _DET_SOURCE = build_detector(model_dir)
    return _DET_SOURCE


def decode_qr(img, model_dir=None):
    """解码 BGR 图像中的第一个二维码文本；未检出返回 None。"""
    global _DET
    if _DET is None:
        configure(model_dir)
    texts, _ = _DET.detectAndDecode(img)
    return texts[0] if texts else None


def detector_source():
    """当前检测器来源："files"（加载了 4 个模型）/ "default"（OpenCV 内建缺省）。"""
    return _DET_SOURCE
