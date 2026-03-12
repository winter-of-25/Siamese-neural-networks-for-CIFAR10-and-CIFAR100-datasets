import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import os
from tqdm import tqdm

def custom_collate(batch):
    # 直接提取 PIL 图像和 Label，跳过复杂的 Tensor 转换，极大减轻 CPU 负担
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, labels

def main():
    print("=== 🚀 RTX 3090 极限压榨版: CIFAR-10 数据集生成 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 开启 FP16 半精度 (3090 必备，速度起飞，显存减半)
    weight_dtype = torch.float16 
    
    print("加载 BLIP 大模型 (FP16 模式)...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large", 
        torch_dtype=weight_dtype
    ).to(device)
    
    # 2. 加载 CIFAR-10 (不使用 Transform，直接获取最原始的 PIL 图像)
    print("准备 CIFAR-10 训练集 (50,000 张)...")
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True)
    classes = train_dataset.classes
    
    # [核心压榨点] Batch Size 拉到 128！
    # 如果 3090 报 OOM (显存不足)，请降到 64。如果显存还有余量，可以试着开到 256！
    batch_size = 128 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8,        # 开启多线程拉取数据
        collate_fn=custom_collate,
        pin_memory=False
    )

    data_records = []
    
    print(f"🔥 开始全速生成！Batch Size: {batch_size}")
    
    # 使用 tqdm 显示进度条
    for images, labels in tqdm(train_loader, desc="Generating Captions"):
        
        # 将一个 Batch 的图片一次性喂给处理器
        inputs = blip_processor(images=images, return_tensors="pt").to(device, weight_dtype)
        
        # 显卡全速推理
        with torch.no_grad():
            # max_new_tokens=20 限制生成长度，避免模型废话，进一步提升速度
            outputs = blip_model.generate(**inputs, max_new_tokens=20)
            
        # 批量解码文本
        captions = blip_processor.batch_decode(outputs, skip_special_tokens=True)
        
        # 组装数据
        for cap, label_idx in zip(captions, labels):
            data_records.append({
                'blip_description': cap.strip(),
                'true_label': classes[label_idx]
            })

    # 3. 保存至指定目录
    output_dir = '../data/cifar10_json'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cifar10_full_train.json')
    
    print("\n💾 正在保存完美版数据集...")
    df = pd.DataFrame(data_records)
    df.to_json(output_path, orient='records', lines=True)
    
    print(f"✅ 大功告成！完美包含了 {len(data_records)} 条无条件语义描述。")
    print(f"文件已保存至: {output_path}")

if __name__ == "__main__":
    main()