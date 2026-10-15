"""WeChatQRCode 解码封装（algorithms 层，仅 live/replay 使用）。

模块级 import cv2 在加载时不打开任何相机/解码器资源；WeChatQRCode 检测器
惰性单例：首次 decode_qr() 才构建（构建较重，避免 sim/测试路径付初始化成本）。
"""
import cv2

_DET = None


def decode_qr(img):
    """解码 BGR 图像中的第一个二维码文本；未检出返回 None。"""
    global _DET
    if _DET is None:
        _DET = cv2.wechat_qrcode_WeChatQRCode()
    texts, _ = _DET.detectAndDecode(img)
    return texts[0] if texts else None
