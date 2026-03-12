import torch

class Config:
    # --- 路径设置 ---
    data_root = '../data'
    json_path = '../data/cifar10_json/cifar10_full_train.json' 
    
    # --- WandB 设置 ---
    use_wandb = True
    wandb_project = "CIFAR10-SOTA-Challenge" 
    wandb_run_name = "WideResNet_BS128_SwinDistill_Enriched"
    
    # --- 硬件设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 16
    
    # --- 数据集参数 ---
    dataset = 'cifar10'  
    num_classes = 10     
    
    batch_size = 128 
    epochs = 200
    
    # 线性缩放定律 + Warmup
    base_lr = 0.05
    lr = base_lr * (batch_size / 128)
    warmup_epochs = 5
    
    # --- Mixup 设置 (修复：补充缺失的 alpha) ---
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