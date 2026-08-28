# UAV Pilot Recognition / 无人机飞手识别

从视频、摄像头或 RTSP 流中识别疑似无人机飞手。算法将人体姿态、人员跟踪、遥控器检测、遥控器与手腕的空间关联以及时序置信度融合为每个人员的最终状态。

## 算法流程

```text
视频 / 摄像头 / RTSP
        ↓
RTMDet person + RTMPose Body26
        ↓
人体质量过滤 + ByteTrackLite 跟踪
        ↓
人体姿态规则（高召回候选）
        ↓
640×640 人体 ROI → RTMDet 遥控器检测
        ↓
遥控器与人体/手腕空间关联
        ↓
姿态 + 遥控器 + 3/5 时序证据融合
        ↓
PERSON / POSSIBLE_PILOT / CONFIRMED_PILOT
```

## 项目结构

```text
uav-pilot-recognition/
├── run.py                                  # 推荐运行入口
├── pilot_controller_video_confidence.py    # 联合检测与置信度融合
├── app.py                                  # 视频、RTMPose、跟踪和输出工具
├── pose_rules_1.py                         # 基础姿态与跟踪规则
├── pose_rules_2.py                         # 当前姿态判定规则
├── final_model2.0/
│   ├── handheld_rtmdet_tiny_960.py         # 遥控器模型配置
│   └── best_model.pth                      # 遥控器模型权重
├── scripts/
│   ├── setup_windows.ps1                   # Windows CUDA 11.8 环境安装
│   └── check_environment.py                # 环境与模型完整性检查
├── tests/test_core.py                      # 不加载大模型的核心单元测试
├── examples/                               # 本地输入，不提交视频
└── outputs/                                # 本地输出，不提交结果
```

这是推理项目，不包含训练数据集、标注图片、测试视频、日志、缓存或训练结果。

## 已验证环境

- Windows，Python 3.8.20
- PyTorch 2.1.2 + CUDA 11.8
- MMCV 2.1.0、MMEngine 0.10.7
- MMDetection 3.3.0、MMPose 1.3.2
- OpenCV 4.10.0、NumPy 1.24/1.26

建议使用 Python 3.8–3.10。PyTorch 和 MMCV 必须按操作系统、显卡和 CUDA 版本匹配安装。

## 安装

### Windows + NVIDIA GPU（CUDA 11.8）

在 PowerShell 中进入项目根目录后执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

也可以手动安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install openmim
python -m mim install "mmcv==2.1.0"
python -m pip install -r requirements.txt
python .\scripts\check_environment.py
```

RTX 50 系显卡需要支持 Blackwell 的新版本 PyTorch/CUDA，并从源码编译带 CUDA 算子的 MMCV；不要直接使用上面的 CUDA 11.8 组合。

## 运行

将待测视频放进 `examples/`，从项目根目录执行：

```powershell
.\.venv\Scripts\python.exe .\run.py `
  --input .\examples\input.mp4 `
  --output .\outputs\result.mp4 `
  --jsonl .\outputs\result.jsonl `
  --device cuda:0 `
  --controller-every 5 `
  --no-show
```

首次运行会由 MMPose 下载 `body26` 所需的人体检测与姿态权重到 `.cache/`，需要联网。控制器权重已包含在 `final_model2.0/` 中。

摄像头：

```powershell
.\.venv\Scripts\python.exe .\run.py --input 0 --device cuda:0
```

RTSP：

```powershell
.\.venv\Scripts\python.exe .\run.py --input "rtsp://USER:PASSWORD@HOST:554/path" --device cuda:0 --no-show
```

不要把包含用户名或密码的 RTSP 地址写进仓库文件、日志或提交记录。

### 常用参数

- `--controller-every 5`：每 5 个姿态帧运行一次遥控器检测，适合约 25 FPS 输入；约 60 FPS 可从 12 开始调整。
- `--controller-batch-size 1`：显存较小时使用 1；显存充足可提高。
- `--frame-stride N`：离线视频每 N 帧推理一次；实时流只允许 1。
- `--no-show`：不打开预览窗口，服务器或批处理建议启用。
- `--max-frames N`：只运行 N 个处理帧，适合快速检查。
- `--device cpu`：CPU 模式，速度会明显慢于 GPU。

查看完整参数：

```powershell
.\.venv\Scripts\python.exe .\run.py --help
```

## 输出

- 标注视频：由 `--output` 指定。系统优先使用 FFmpeg 输出 H.264；找不到 FFmpeg 时回退到 OpenCV MP4V。
- JSONL：由 `--jsonl` 指定，每行对应一个处理帧，包含人体框、Body26 关键点、轨迹 ID、姿态证据、遥控器框、时序票数和融合置信度。
- `PERSON`：普通人员。
- `POSSIBLE_PILOT`：姿态可疑或遥控器证据尚未稳定。
- `CONFIRMED_PILOT`：多帧融合后确认的飞手。

## 验证

```powershell
.\.venv\Scripts\python.exe .\scripts\check_environment.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

环境检查会验证 Python 包、MMCV CUDA/CPU 算子、控制器配置、权重文件及权重 SHA-256。单元测试不会加载两个大模型。

## 上传 GitHub 前

仓库包含约 43.7 MB 的 `best_model.pth`。GitHub 可以接收该大小的单个文件，但会给出“大文件”提示；若模型还会继续变大，建议改用 Git LFS 或 GitHub Release。请确认你拥有发布该权重和源代码的权利。

项目当前未附带开源许可证；公开发布前，请根据代码和模型的实际授权方式选择合适的许可证。
