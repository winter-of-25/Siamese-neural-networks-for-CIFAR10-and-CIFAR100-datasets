import torch

class Config:
    # --- 路径设置 ---
    data_root = '../data'
    # 指向 4090 刚刚新鲜生成的 100 类文本数据集
    json_path = '../data/cifar100_json/cifar100_full_train.json' 
    
    # --- WandB 设置 ---
    use_wandb = True
    wandb_project = "CIFAR100-SOTA-Challenge" # 切换到新项目面板
    wandb_run_name = "WideResNet_BS256_SwinDistill_Enriched"
    
    # --- 硬件设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 16
    
    # --- 数据集参数 ---
    dataset = 'cifar100'  # 切换
    num_classes = 100     # 切换
    
    # 4090 (24G) 的极限安全区，不可贪心
    batch_size = 256
    epochs = 300
    
    # 线性缩放定律 + Warmup (自动根据 BS 计算，无需手动算)
    base_lr = 0.05
    lr = base_lr * (batch_size / 128)
    warmup_epochs = 5
    
    # --- Mixup 设置 ---
    alpha = 1.0 
    
    # SAM & Loss 权重
    rho = 0.05
    weight_decay = 5e-4
    w_ce = 1.0   
    w_kd = 2.0   
    w_sim = 1.0  
    T = 4.0      
    
    # Text Encoder
    vocab_size = 6000
    max_len = 30
    embed_dim = 128