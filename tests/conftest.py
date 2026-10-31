"""pytest 全局夹具：把工作目录锚定到仓库根。

项目里大量测试与工具用**相对路径**引用 ``config/default.yaml`` /
``config/field.example.yaml`` / ``tests/fixtures``（这是 README 明示的用法：
在仓库根运行）。此前这套约定只靠「人在根目录敲 pytest」保证，一旦从别处
调用（CI、IDE、`pytest /path/to/repo`）就会 FileNotFoundError。

这里在收集开始时统一切到仓库根并还原，既保留既有相对路径写法，又让测试
在任何调用目录下都成立。
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(*parts):
    """仓库内绝对路径（不受调用目录影响）。"""
    return os.path.join(REPO_ROOT, *parts)


# 供测试模块在**导入期**使用：把仓库根加入 sys.path，使 `import src...` 与
# `from tools... import ...` 在任何调用目录下都可用（不依赖 pytest 的 pythonpath
# 配置项——该选项需 pytest>=7，目标机 JetPack 自带的是 6.2.5）。
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture(scope="session", autouse=True)
def _anchor_cwd_to_repo_root():
    prev = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield REPO_ROOT
    finally:
        os.chdir(prev)
