# Jetson 部署记录（实测环境，2026-10-31）

目标机：`uav@192.168.137.88`（SSH，密码见组内文档，勿提交到仓库）

## 1. 机器规格（实测）

| 项 | 值 |
|---|---|
| 型号 | Jetson Orin 系列（`uname -m` = aarch64，内核 `5.15.148-tegra`） |
| L4T | R36.4.4（JetPack 6.2） |
| 系统 | Ubuntu 22.04.5 LTS |
| Python | 3.10.12（系统自带，满足项目 `>=3.10`） |
| OpenCV | **4.5.4，JetPack 自带，已含 `cv2.wechat_qrcode_WeChatQRCode`** |
| 内存/磁盘 | 充足（本项目仅需 ~100KB 源码 + 依赖） |

关键结论：**不要 `pip install opencv-python`**。JetPack 自带的 `python3-opencv`
是 NVIDIA 编译版本且自带 WeChatQRCode 模块，覆盖安装会破坏 CUDA 绑定。

## 2. 已完成的安装步骤

```bash
# pip（Ubuntu 22.04 默认没有）
python3 /tmp/get-pip.py --user

# 项目依赖（只补缺的，cv2 不动）
python3 -m pip install --user pyserial pyyaml

# pytest 升到 8.x（6.2.5 不认 pytest.ini 的 pythonpath 选项；
# 代码已在 conftest.py 里做了 sys.path 兜底，故 6.2.5 也能跑）
python3 -m pip install --user "pytest>=7,<9"
```

已装版本：pyserial 3.5、pyyaml 5.4.1、pytest 8.4.2、numpy 1.21.5、opencv 4.5.4。

## 3. 部署与验证

代码位于 `/home/uav/gongchuang-vision`。上传方式为打包源码（排除 `.git`、
`__pycache__`、`logs`、`图片和附件/`）后 SFTP 传输并解包。

```bash
cd /home/uav/gongchuang-vision

python3 -m pytest -q                 # 85 passed（在任意 cwd 下均可）
python3 -m src.main --mode sim       # 全流程到 DONE，退出码 0
```

实测结果：
- pytest 8.4.2：**85 passed**（项目根与 `/tmp` 两种 cwd 均通过）
- pytest 6.2.5：**85 passed**（仅 `pythonpath` 选项警告，已由 conftest 兜底）
- `--mode sim`：`IDLE → … → DONE`，两次 `CAL_APPLIED`，退出码 0

## 4. 尚未完成（上赛场前必须做）

1. **WeChatQRCode 模型文件未部署**：机器上没有任何 `.prototxt/.caffemodel`
   （`/usr/share/opencv4` 下也没有）。当前 `decode_qr` 会退回无参构造，
   使用 OpenCV 内置缺省模型——**辨识率明显低于加载模型**。需从
   https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode 下载
   `detect.prototxt` / `detect.caffemodel` / `sr.prototxt` / `sr.caffemodel`，
   放到项目内（如 `models/wechat_qr/`），并设置 `camera.qr_model_dir`。
2. **无相机设备**：`/dev/video*` 当前为空（未插相机），`live` 模式无法验证。
3. **相机外参未标定**：`camera_calib_file` 指向的仍是示例文件。现场需按
   README §5 用 `tools/calibrate_camera.py` 实测标定。
4. **串口未接**：`serial.port` 默认 `COM3`（Windows 命名），Linux 下应为
   `/dev/ttyTHS1` 或 `/dev/ttyUSB0` 之类，需按实际主控连接确认。
5. **反向隧道未使用**：机器可直连 PyPI（已配 USTC 镜像），
   `ssh -R 7890:127.0.0.1:7897` 在本场景非必需。
