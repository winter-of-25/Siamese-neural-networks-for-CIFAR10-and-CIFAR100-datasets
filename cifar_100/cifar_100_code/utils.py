import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import re
from collections import Counter
from torch.utils.data import Dataset
from torchvision import datasets
from torch.optim.lr_scheduler import _LRScheduler

# === 1. 简易 Tokenizer (兼容 JSONL) ===
class SimpleTokenizer:
    def __init__(self, json_path, vocab_size=5000, max_len=30):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.max_len = max_len
        print(f"Building Vocab from {json_path}...")
        
        all_text = ""
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    text = item.get('blip_description') or item.get('caption') or item.get('text', "")
                    all_text += " " + str(text).lower()
            
            words = re.findall(r'\w+', all_text)
            counts = Counter(words)
            for word, _ in counts.most_common(vocab_size - 4):
                self.vocab[word] = len(self.vocab)
            print(f"✅ Vocab built! Size: {len(self.vocab)}")
        else:
            print(f"⚠️ WARNING: JSON not found at {json_path}. Using DUMMY vocab.")
            
    def encode(self, text):
        text = str(text).lower()
        words = re.findall(r'\w+', text)
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        ids = ids[:self.max_len-2]
        ids = [self.vocab["<SOS>"]] + ids + [self.vocab["<EOS>"]]
        if len(ids) < self.max_len:
            ids += [self.vocab["<PAD>"]] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

# === 2. 真实语义数据集 (按类别池化采样，增强泛化能力) ===
class RealEnrichedDataset(Dataset):
    def __init__(self, root, json_path, train=True, transform=None, tokenizer=None):
        # 核心修改：改为 CIFAR100
        self.cifar = datasets.CIFAR100(root=root, train=train, download=True, transform=None)
        self.transform = transform
        self.tokenizer = tokenizer
        
        # 建立类别语义池 { "cat": ["desc1", "desc2"...] }
        self.label_to_captions = {}
        if train and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    cap = item.get('blip_description', "")
                    label = item.get('true_label', "")
                    
                    if label and cap:
                        if label not in self.label_to_captions:
                            self.label_to_captions[label] = []
                        self.label_to_captions[label].append(cap)
                        
        self.classes = self.cifar.classes
        
    def __getitem__(self, index):
        img, target = self.cifar[index]
        if self.transform:
            img = self.transform(img)
            
        class_name = self.classes[target]
        captions = self.label_to_captions.get(class_name)
        
        if captions:
            import random
            caption_text = random.choice(captions)
        else:
            caption_text = f"a photo of a {class_name}"
            
        if self.tokenizer:
            caption_ids = self.tokenizer.encode(caption_text)
        else:
            caption_ids = torch.zeros(30, dtype=torch.long)
            
        return img, target, caption_ids

    def __len__(self):
        return len(self.cifar)

# === 3, 4, 5. 调度器、SAM、辅助函数 (保持完美原样) ===
class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=0.001, eta_min=0.0):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(WarmupCosineLR, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr) for base_lr in self.base_lrs]
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + np.cos(np.pi * progress)) for base_lr in self.base_lrs]

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["e_w"] = p.grad * scale.to(p)
                p.add_(self.state[p]["e_w"]) 
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(p.device)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]), p=2)
        return norm

def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def distillation_loss(student_logits, teacher_logits, T):
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    return F.kl_div(student_log_probs, soft_targets, reduction='batchmean') * (T * T)