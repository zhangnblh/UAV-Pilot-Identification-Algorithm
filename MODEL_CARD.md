# 遥控器检测模型说明

- 模型：单类别 RTMDet-Tiny，类别为 `controller`
- 输入：算法会把每个人体框扩展、补边并缩放到 `640 × 640` 后送入模型
- 配置：`final_model2.0/handheld_rtmdet_tiny_960.py`
- 权重：`final_model2.0/best_model.pth`
- 权重大小：43,674,013 bytes
- SHA-256：`9efffdbb7ea60572a61af21da47688dce9d319f7e848435e61476fccecb4015f`

该仓库中的权重只用于推理。训练数据集、数据标注和训练过程产物未包含在此推理项目中。

算法还使用 MMPose 的 `body26` 预训练人体检测/姿态权重。它们会在首次运行时由 MMPose 自动下载到项目的 `.cache/` 目录，因此 `.cache/` 不应提交到 GitHub。
