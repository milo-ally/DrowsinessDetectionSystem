# DDD - 疲劳检测系统

DDD (Drowsiness Detection System) 是一个基于深度学习的实时疲劳检测系统，能够通过摄像头实时检测人脸并判断驾驶员的疲劳状态。

## 项目简介

本项目提供了两种实现版本，分别适用于不同的部署场景：

- **PyTorch 版本**: 适用于通用 PC/服务器环境，使用 PyTorch 框架进行推理
- **RKNN 版本**: 适用于 RKNN 设备（如 RK3588），使用 RKNN 工具链进行模型优化和部署

系统采用两阶段检测流程：
1. **人脸检测**: 使用 YOLO 模型实时检测视频流中的人脸位置
2. **疲劳分类**: 对检测到的人脸区域进行疲劳状态分类（Drowsy/NonDrowsy）

## 功能特性

- ✅ 实时人脸检测与疲劳状态识别
- ✅ 支持摄像头实时视频流处理
- ✅ 高精度疲劳状态分类（Drowsy/NonDrowsy）
- ✅ 可视化检测结果（标注框和状态标签）
- ✅ 支持 GPU 加速（PyTorch 版本）
- ✅ 针对边缘设备优化的 RKNN 版本

## 项目结构

```
DDD-app/
├── pytorch/              # PyTorch 版本实现
│   ├── app.py           # 应用入口
│   ├── core/
│   │   ├── 1_detect.py  # 人脸检测模块
│   │   ├── 2_classify.py # 疲劳分类模块
│   │   ├── 3_final.py   # 集成版本（检测+分类）
│   │   ├── models/      # 模型定义
│   │   └── weights/     # 模型权重文件
│   └── reqirements.txt  # Python 依赖
│
└── rknn/                 # RKNN 版本实现
    ├── app.py           # 应用入口
    ├── core/
    │   ├── 1_classification_test.py  # 分类测试
    │   ├── 2_face_detection.py       # 人脸检测
    │   ├── 3_face_detection_camera.py # 摄像头检测
    │   ├── 4_final_test.py           # 最终集成测试
    │   ├── test_images/              # 测试图片
    │   └── weights/                  # ONNX/RKNN 模型文件
    └── requirements.txt              # Python 依赖
```

## 技术栈

### PyTorch 版本
- **深度学习框架**: PyTorch
- **计算机视觉**: OpenCV, PIL
- **模型架构**: YOLO (人脸检测) + 自定义分类网络

### RKNN 版本
- **推理框架**: RKNN Toolkit2
- **模型格式**: ONNX → RKNN
- **目标平台**: RK3588 等 RKNN 设备

## 依赖要求

### PyTorch 版本
- Python 3.x
- PyTorch
- torchvision
- OpenCV
- NumPy
- Pillow

### RKNN 版本
- Python 3.10
- RKNN Toolkit2
- ONNX / ONNXRuntime
- OpenCV
- NumPy

## 模型说明

系统使用两个预训练模型：
- **人脸检测模型** (`detect.pt` / `face.onnx`): 基于 YOLO 架构，用于检测视频帧中的人脸位置
- **疲劳分类模型** (`classify.pt` / `face_classification.onnx`): 用于对检测到的人脸进行疲劳状态分类

## 应用场景

- 🚗 驾驶员疲劳监测系统
- 👨‍💼 办公场景疲劳提醒
- 📹 视频监控中的疲劳检测
- 🔒 安全关键场景的状态监控

## 注意事项

- 模型权重文件需要单独下载并放置在对应的 `weights/` 目录下
- RKNN 版本需要配置正确的摄像头设备路径（默认 `/dev/video11`）
- 建议在光线充足的环境下使用以获得最佳检测效果

