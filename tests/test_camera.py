import numpy as np
from src.devices.camera import ImageFolder, make_camera

def test_image_folder_sequence(tmp_path):
    import subprocess, sys
    subprocess.run([sys.executable, "tests/fixtures/gen_images.py"], check=True)
    cam = ImageFolder("tests/fixtures/imgs")
    frames = []
    while True:
        ok, f = cam.read()
        if not ok:
            break
        frames.append(f)
    assert len(frames) == 3 and frames[0].shape == (480, 640, 3)

def test_make_camera_replay():
    cam = make_camera({"replay_path": "tests/fixtures/imgs"}, "replay")
    assert isinstance(cam, ImageFolder)


# 以下为结构性补充：不触碰真实摄像头/视频文件解码，仅验证类型与工厂分流。

def test_image_folder_loop():
    cam = ImageFolder("tests/fixtures/imgs", loop=True)
    ok1, _ = cam.read()
    for _ in range(2):
        cam.read()
    ok2, f = cam.read()  # 越过末尾后回到 frame_000
    assert ok1 and ok2 and f.shape == (480, 640, 3)

def test_usb_and_video_structural():
    from src.devices.camera import USBCamera, VideoFile, CameraSource
    # VideoFile 对不存在文件只置 isOpened()=False，不抛异常；不打开 index=0 以免接触真实相机
    vf = VideoFile("tests/fixtures/missing.mp4")
    ok, frame = vf.read()
    assert ok is False and frame is None
    vf.release()
    for cls in (USBCamera, VideoFile):
        assert hasattr(cls, "read") and hasattr(cls, "release")
    assert hasattr(ImageFolder, "read")
    assert isinstance(vf, CameraSource) and isinstance(ImageFolder("tests/fixtures/imgs"), CameraSource)

def test_make_camera_replay_video_and_live_keys():
    from src.devices.camera import VideoFile
    cam = make_camera({"replay_path": "tests/fixtures/missing.mp4"}, "replay")
    assert isinstance(cam, VideoFile)
    cam.release()
    # sim 模式循环读取 fixture 图
    sim = make_camera({"sim_images": "tests/fixtures/imgs"}, "sim")
    assert isinstance(sim, ImageFolder)
