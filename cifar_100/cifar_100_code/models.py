import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# === 1. 学生视觉编码器 (WideResNet) ===
class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out

class StudentWideResNet(nn.Module):
    # 修复：默认 num_classes 改为 10
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10, embed_dim=128):
        super(StudentWideResNet, self).__init__()
        self.in_planes = 16
        n = (depth - 4) // 6
        k = widen_factor
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._wide_layer(WideBasic, 16*k, n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, 32*k, n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, 64*k, n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(64*k)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.linear = nn.Linear(64*k, num_classes)
        self.projector = nn.Sequential(
            nn.Linear(64*k, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1)
        
        logits = self.linear(feat)
        img_emb = self.projector(feat)
        return logits, F.normalize(img_emb, p=2, dim=1)

# === 2. 学生文本编码器 ===
class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super(SimpleTextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.proj = nn.Linear(256, embed_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1) 
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)

# === 3. 孪生包装器 ===
class SiameseNetwork(nn.Module):
    def __init__(self, vision_model, text_model):
        super(SiameseNetwork, self).__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        
    def forward(self, img, text_ids):
        logits, img_emb = self.vision_model(img)
        text_emb = self.text_model(text_ids)
        return logits, img_emb, text_emb

# === 4. 老师模型 (Swin) ===

def get_teacher_model(num_classes=10, device='cuda'):
    print(">>> Loading Teacher: Swin-Transformer (ImageNet Pretrained)...")
    
    # 修复核心：直接在 create_model 传入 num_classes
    # 这样 timm 会自动保留底层的 Pooling 逻辑，安全替换最终的分类器！
    teacher = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
    
    # 冻结参数
    for param in teacher.parameters():
        param.requires_grad = False
        
    teacher.to(device)
    teacher.eval()
    return teacher