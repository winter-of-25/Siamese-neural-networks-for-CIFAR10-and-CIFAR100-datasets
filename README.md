# 🚀 Siamese Neural Networks for CIFAR-10 and CIFAR-100 Datasets

本仓库实现了一种**非对称多模态知识蒸馏框架 (Asymmetric Multimodal Distillation Framework)**，旨在在不增加任何推理成本的前提下，利用大语言模型（LLMs）和强大的视觉预训练模型提升轻量级 CNN（WideResNet）的图像分类表现。

## 💡 核心创新点 (Key Innovations)

1. **离线多模态语义富集 (Offline Semantic Enrichment)**
* 摒弃了在训练和推理时挂载庞大文本模型的传统做法。使用 **BLIP (Salesforce/blip-image-captioning-large)** 在离线阶段对 CIFAR 图像进行“看图说话”，生成富含上下文的文本描述（JSONL格式）。
* 训练时采用**动态语义池采样**，同一类别的图像在不同 Epoch 会匹配不同的描述，实现跨模态的高阶数据增强。


2. **孪生知识蒸馏架构 (Siamese Knowledge Distillation)**
* **Teacher**: 冻结的在 ImageNet 上预训练的 Swin Transformer (`swin_tiny_patch4_window7_224`)，提供软标签指导 (Soft Targets)。
* **Student**: 完全从头训练 (Train from scratch) 的 WideResNet28-10。
* **Text Encoder**: 从头训练的极简两层 Transformer。通过余弦嵌入损失 (Cosine Embedding Loss) 强制视觉特征与文本特征在统一的高维空间中对齐。


3. **零额外开销推理 (Zero-cost Inference)**
* 训练结束后，Swin 老师和文本编码器被完全丢弃。部署和测试时**仅保留轻量级的 WideResNet**，在实现极高准确率的同时保持极致的推理速度。


4. **带有防爆机制的 SAM 优化引擎 (Robust SAM Optimizer)**
* 针对 CIFAR-100 数据量少易过拟合的痛点，引入 **SAM (Sharpness-Aware Minimization)** 寻找平坦极小值。
* 针对 FP16 混合精度 (AMP) 下 SAM 极易产生 `NaN` 梯度爆炸的问题，在核心训练循环中实现了**自定义梯度裁剪与状态解耦**的修复方案。



---

## 📁 仓库结构 (Repository Structure)

```text
Siamese-neural-networks-for-CIFAR10-and-CIFAR100-datasets/
├── cifar_10/                           # CIFAR-10 训练工作区
│   └── cifar_10_code/
│       ├── config.py                   # 超参配置中心
│       ├── generate_data.py            # BLIP 离线数据生成脚本
│       ├── main.py                     # 核心训练循环 (含 SAM+AMP)
│       ├── models.py                   # 孪生网络与 Swin 老师架构
│       └── utils.py                    # 动态语义池、Tokenizer 与 Loss
├── cifar_100/                          # CIFAR-100 训练工作区
│   └── cifar_100_code/
│       ├── config.py
│       ├── generate_cifar100.py        # 包含 Swin+BLIP 双重提取的增强脚本
│       ├── main.py
│       ├── models.py
│       └── utils.py
├── LICENSE                             # Apache 2.0 开源协议
└── README.md                           # 项目说明文档

```

---

## ⚙️ 环境依赖 (Dependencies)

建议使用 Python 3.8+ 及 PyTorch 2.0+（支持 `torch.compile` 动态编译加速）。

```bash
pip install torch torchvision torchaudio
pip install transformers timm datasets
pip install pandas tqdm wandb

```

---

## 🚀 快速开始 (Quick Start)

### 步骤 1: 离线生成富集语义数据 (Offline Data Generation)

首先利用强大的 GPU（如 RTX 3090/4090）通过 BLIP 模型生成无条件的文本描述数据集。

对于 CIFAR-100：

```bash
cd cifar_100/cifar_100_code
python generate_cifar100.py

```

*(数据将自动保存至 `../data/cifar100_json/cifar100_full_train.json`)*

### 步骤 2: 启动孪生网络训练 (Training)

代码集成了 Weights & Biases (WandB) 用于实时监控训练曲线，以及自动断点保存机制（每 50 轮强制存档）。

```bash
# 请先在 config.py 中根据硬件配置调整 batch_size 和 epochs
python main.py

```

**训练时的 Loss 组合机理：**


$$\mathcal{L}_{total} = w_{ce} \mathcal{L}_{ce} + w_{kd} \mathcal{L}_{kd} + w_{sim} \mathcal{L}_{sim}$$

* $\mathcal{L}_{ce}$: Mixup 增强后的交叉熵损失 (Hard Label)。
* $\mathcal{L}_{kd}$: 约束学生模仿 Swin 老师概率分布的 KL 散度损失 (Soft Label)。
* $\mathcal{L}_{sim}$: 强制视觉特征与文本编码特征拉齐的余弦嵌入损失。

### 步骤 3: 评估与推理 (Evaluation)

验证集的评估代码已内嵌在 `main.py` 中。在 `validate` 函数中，系统会自动执行**模型物理剥离**，仅向视觉模型传入图像张量，彻底验证 Zero-cost Inference 状态下的准确率。

---

## 📈 超参数推荐 (Recommended Configurations)

| 参数 (Hyperparameter) | CIFAR-10 | CIFAR-100 | 说明 (Description) |
| --- | --- | --- | --- |
| `batch_size` | 128 ~ 512 | 256 | 根据显存动态调整 |
| `epochs` | 200 | 300 | 配合 Cosine 退火策略 |
| `alpha` (Mixup) | 1.0 | 1.0 | 控制流形空间混合强度 |
| `rho` (SAM) | 0.05 | 0.05 | 邻域扰动半径 |
| `w_ce`, `w_kd`, `w_sim` | 1.0, 2.0, 1.0 | 1.0, 2.0, 1.0 | 交叉熵、蒸馏、对齐的权重分配 |
| `T` (Temperature) | 4.0 | 4.0 | 知识蒸馏的平滑温度 |

---

## 📄 开源协议 (License)

本项目遵循 [Apache License 2.0](https://www.google.com/search?q=LICENSE) 开源协议。允许用于商业用途、修改、分发及私人使用。
