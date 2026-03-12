import os
import json
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

def main():
    output_dir = '../data/cifar100_json'
    os.makedirs(output_dir, exist_ok=True)
    
    train_output_path = os.path.join(output_dir, 'cifar100_full_train.json')
    test_output_path = os.path.join(output_dir, 'cifar100_full_test.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 1. 加载模型 ===
    print("🚀 Loading Swin & BLIP Models to 4090...")
    processor = AutoImageProcessor.from_pretrained("MazenAmria/swin-base-finetuned-cifar100")
    model = AutoModelForImageClassification.from_pretrained("MazenAmria/swin-base-finetuned-cifar100").to(device)

    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

    # === 2. 取消 Streaming，全量加载进内存（CIFAR很小，这样能大幅加速） ===
    print("⏳ Downloading & Loading CIFAR-100 into RAM...")
    train_dataset = load_dataset("uoft-cs/cifar100", split="train")
    test_dataset = load_dataset("uoft-cs/cifar100", split="test")
    
    label_feature = train_dataset.features['fine_label']

    # === 3. 数据打包器 (Collate Function) ===
    def collate_fn(batch):
        images = [item['img'].convert("RGB") for item in batch]
        labels = [item['fine_label'] for item in batch]
        return images, labels

    # === 4. 批处理引擎 (Batch Processing) ===
    def process_dataset_batched(dataset, output_path, desc, batch_size=128):
        # 开启多线程帮厨喂数据
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=8, pin_memory=True)
        
        with open(output_path, 'w', encoding='utf-8') as f_output:
            pbar = tqdm(total=len(dataset), desc=desc, dynamic_ncols=True)

            for images, labels in loader:
                # --- 批量处理 Swin 预测 ---
                inputs_swin = processor(images=images, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs_swin)
                    predicted_class_idxs = outputs.logits.argmax(dim=-1).tolist()
                    swin_class_names = [label_feature.int2str(idx) for idx in predicted_class_idxs]

                # --- 批量处理 BLIP 生成 (核心提速点) ---
                inputs_blip = blip_processor(images=images, return_tensors="pt").to(device)
                with torch.no_grad():
                    out_blip = blip_model.generate(**inputs_blip)
                blip_descriptions = blip_processor.batch_decode(out_blip, skip_special_tokens=True)

                # --- 批量写入 JSONL ---
                for swin_name, blip_desc, true_label_idx in zip(swin_class_names, blip_descriptions, labels):
                    true_class_name = label_feature.int2str(true_label_idx)
                    data_entry = {
                        'swin_description': f"This is a photo of a {swin_name}.",
                        'blip_description': blip_desc,
                        'true_label': true_class_name
                    }
                    f_output.write(json.dumps(data_entry) + '\n')
                
                pbar.update(len(images))

            pbar.close()

    # === 5. 满血开跑 ===
    # 4090 显存 24G，Batch Size 开到 128 毫无压力
    print("🔥 Start processing training dataset (Batch Size 128)...")
    process_dataset_batched(train_dataset, train_output_path, desc="Train Images", batch_size=128)

    print("🔥 Start processing test dataset...")
    process_dataset_batched(test_dataset, test_output_path, desc="Test Images", batch_size=128)

    print("✅ All done! 4090 inference complete.")

if __name__ == "__main__":
    main()