import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time
import os
import wandb
import torch._dynamo 
import timm 

from config import Config
from models import StudentWideResNet, SimpleTextEncoder, SiameseNetwork
from utils import SimpleTokenizer, RealEnrichedDataset, SAM, WarmupCosineLR, mixup_data, distillation_loss

def main():
    print(f"=== 🚀 SOTA CIFAR-100 Training on 4090 ===")
    
    save_dir = './checkpoints_cifar100'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Models will be saved to: {save_dir}")

    if Config.use_wandb:
        wandb.init(
            project=Config.wandb_project,
            name=Config.wandb_run_name,
            config={
                "batch_size": Config.batch_size,
                "lr": Config.lr,
                "epochs": Config.epochs,
                "strategy": "Swin_Distill + Text_Alignment + SAM + AMP_Fix"
            }
        )
    
    torch.backends.cudnn.benchmark = True
    
    tokenizer = SimpleTokenizer(Config.json_path, Config.vocab_size, Config.max_len)
    
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    # 注意：AutoAugmentPolicy.CIFAR10 在学术界通用，无需修改
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    print("Loading Data...")
    train_set = RealEnrichedDataset(root=Config.data_root, json_path=Config.json_path, train=True, transform=train_transform, tokenizer=tokenizer)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=Config.num_workers, 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=4
    )
    
    # 核心修改：测试集也必须替换为 CIFAR100
    test_set = datasets.CIFAR100(root=Config.data_root, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=4)
    
    vision_net = StudentWideResNet(num_classes=Config.num_classes, embed_dim=Config.embed_dim)
    text_net = SimpleTextEncoder(vocab_size=Config.vocab_size, embed_dim=Config.embed_dim)
    
    student = SiameseNetwork(vision_net, text_net).to(Config.device, memory_format=torch.channels_last)
    
    print(">>> Loading Teacher: Swin-Transformer (ImageNet Pretrained)...")
    teacher = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=Config.num_classes)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher = teacher.to(Config.device, memory_format=torch.channels_last)
    teacher.eval()
    
    torch._dynamo.config.suppress_errors = True 
    try:
        print("Compiling Vision Model (Torch 2.0)...")
        student.vision_model = torch.compile(student.vision_model)
    except Exception as e:
        print(f"Compile skipped: {e}")

    base_optimizer = torch.optim.SGD
    optimizer = SAM(student.parameters(), base_optimizer, rho=Config.rho, lr=Config.lr, momentum=0.9, weight_decay=Config.weight_decay)
    scheduler = WarmupCosineLR(optimizer.base_optimizer, warmup_epochs=Config.warmup_epochs, max_epochs=Config.epochs)
    
    scaler = torch.cuda.amp.GradScaler()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_sim = nn.CosineEmbeddingLoss()

    best_acc = 0.0

    for epoch in range(Config.epochs):
        student.train()
        total_loss, correct, total = 0, 0, 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets, caption_ids) in enumerate(train_loader):
            inputs = inputs.to(Config.device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(Config.device, non_blocking=True)
            caption_ids = caption_ids.to(Config.device, non_blocking=True)
            
            with torch.no_grad():
                t_inputs = F.interpolate(inputs, size=(224, 224), mode='bicubic', align_corners=False)
                t_inputs = t_inputs.contiguous(memory_format=torch.channels_last)
                teacher_logits = teacher(t_inputs)
            
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, Config.alpha, Config.device)
            
            # ================== SAM Step 1 ==================
            with torch.cuda.amp.autocast():
                logits, img_emb, text_emb = student(inputs, caption_ids)
                loss_ce = lam * criterion_ce(logits, targets_a) + (1 - lam) * criterion_ce(logits, targets_b)
                loss_kd = distillation_loss(logits, teacher_logits, Config.T)
                loss_sim = criterion_sim(img_emb, text_emb, torch.ones(inputs.size(0)).to(Config.device))
                loss = Config.w_ce*loss_ce + Config.w_kd*loss_kd + Config.w_sim*loss_sim

            scaler.scale(loss).backward()
            optimizer.first_step(zero_grad=True)
            
            # ================== SAM Step 2 ==================
            with torch.cuda.amp.autocast():
                logits_2, img_emb_2, text_emb_2 = student(inputs, caption_ids)
                loss_ce_2 = lam * criterion_ce(logits_2, targets_a) + (1 - lam) * criterion_ce(logits_2, targets_b)
                loss_kd_2 = distillation_loss(logits_2, teacher_logits, Config.T)
                loss_sim_2 = criterion_sim(img_emb_2, text_emb_2, torch.ones(inputs.size(0)).to(Config.device))
                loss_2 = Config.w_ce*loss_ce_2 + Config.w_kd*loss_kd_2 + Config.w_sim*loss_sim_2
                
            scaler.scale(loss_2).backward()
            
            with torch.no_grad(): 
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is not None and "e_w" in optimizer.state[p]:
                            p.sub_(optimizer.state[p]["e_w"])
                        
            scaler.step(optimizer.base_optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            
            if Config.use_wandb and batch_idx % 10 == 0:
                wandb.log({"train/batch_loss": loss.item()})
            
        scheduler.step()
        
        epoch_time = time.time() - start_time
        ips = total / epoch_time
        curr_lr = optimizer.param_groups[0]['lr']
        train_acc = 100.*correct/total
        avg_loss = total_loss/(batch_idx+1)
        
        print(f"Epoch {epoch+1}/{Config.epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | LR: {curr_lr:.4f} | Speed: {ips:.0f} img/s")
        
        if Config.use_wandb:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/epoch_acc": train_acc,
                "train/learning_rate": curr_lr,
                "epoch": epoch+1
            })
        
        if (epoch + 1) % 5 == 0 or epoch > (Config.epochs - 20):
            val_acc = validate(student, test_loader)
            if Config.use_wandb:
                wandb.log({"val/accuracy": val_acc, "epoch": epoch+1})
            
            if val_acc > best_acc:
                print(f"🔥 New Best Accuracy: {val_acc:.2f}% (was {best_acc:.2f}%) --> Saving model...")
                best_acc = val_acc
                torch.save(student.state_dict(), os.path.join(save_dir, "best_model.pth"))
                with open(os.path.join(save_dir, "best_score.txt"), "w") as f:
                    f.write(f"Best Acc: {best_acc:.2f}% at Epoch {epoch+1}")
        
        if (epoch + 1) % 50 == 0:
            periodic_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            torch.save(student.state_dict(), periodic_path)
            print(f"✅ Periodic Model Saved: {periodic_path}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(save_dir, "last_checkpoint.pth"))

    if Config.use_wandb:
        wandb.finish()
    
    print(f"Training Finished! Best Accuracy: {best_acc:.2f}%")

def validate(model, loader):
    model.eval()
    correct = 0
    total = 0
    vision_model = model.vision_model if hasattr(model, 'vision_model') else model
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(Config.device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(Config.device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits, _ = vision_model(inputs)
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    print(f"--- Val Accuracy: {acc:.2f}% ---")
    return acc

if __name__ == '__main__':
    main()