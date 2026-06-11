# CIFAR-10/CIFAR-100 孪生知识蒸馏网络

本项目实现面向 CIFAR-10 和 CIFAR-100 的多模态知识蒸馏训练流程。训练阶段结合 WideResNet 学生网络、Swin Transformer 教师网络和轻量文本编码器；推理阶段仅保留学生网络。

## 已实现功能

- WideResNet-28-10 图像分类学生模型
- 预训练 Swin Transformer 教师模型
- 轻量 Transformer 文本编码器
- 图像与文本特征余弦对齐和 Soft Target 蒸馏
- Mixup、AutoAugment、SAM 和混合精度训练
- Warmup + Cosine 学习率调度
- W&B 实验记录和定期检查点保存
- 使用 BLIP 生成 CIFAR 图像文本描述
- 独立的 CIFAR-10 与 CIFAR-100 工作区

## 技术栈

- Python、PyTorch、torchvision
- timm、Transformers
- Hugging Face Datasets
- pandas、tqdm、Weights & Biases

## 目录结构

```text
.
├── cifar_10/cifar_10_code/
│   ├── config.py
│   ├── generate_data.py
│   ├── main.py
│   ├── models.py
│   └── utils.py
├── cifar_100/cifar_100_code/
│   ├── config.py
│   ├── generate_cifar100.py
│   ├── main.py
│   ├── models.py
│   └── utils.py
├── LICENSE
└── README.md
```

## 快速开始

```bash
git clone https://github.com/winter-of-25/Siamese-neural-networks-for-CIFAR10-and-CIFAR100-datasets.git
cd Siamese-neural-networks-for-CIFAR10-and-CIFAR100-datasets
pip install torch torchvision torchaudio
pip install timm transformers datasets pandas tqdm wandb
```

CIFAR-10：

```bash
cd cifar_10/cifar_10_code
python generate_data.py
python main.py
```

CIFAR-100：

```bash
cd cifar_100/cifar_100_code
python generate_cifar100.py
python main.py
```

运行前请检查对应 `config.py` 中的数据路径、批量大小、训练轮数、工作进程数、CUDA 设备和 W&B 设置。

## 注意事项

- 仓库未提供 `requirements.txt`，依赖需按代码导入安装。
- 数据生成脚本会从 Hugging Face 下载数据和预训练模型，需要网络及较大的缓存空间。
- BLIP 和 Swin 显存占用较高，显存不足时应降低批量大小。
- 默认训练配置偏向 CUDA 环境；CPU 虽可被自动选中，但完整训练耗时较长。
- 开启 W&B 时需要提前登录；不使用时可在 `config.py` 中关闭。
- README 不声明未经固定环境复现实验验证的准确率。

## 许可证

本项目使用 [Apache License 2.0](./LICENSE)。

## 联系方式

- GitHub: [winter-of-25](https://github.com/winter-of-25)
- Email: [A3762577373@outlook.com](mailto:A3762577373@outlook.com)
