import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class IrisMatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

        # 去掉 classifier，只保留特征提取部分
        self.backbone = mobilenet.features  # 输出 shape: (B, 960, H, W)

        # 添加 global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出 shape: (B, 960, 1, 1)

        self.mlp = nn.Sequential(
            nn.Linear(960 * 3, 1280),
            nn.ReLU(),
            nn.Linear(1280, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        f1 = self.pool(self.backbone(img1)).squeeze(-1).squeeze(-1)  # (B, 960)
        f2 = self.pool(self.backbone(img2)).squeeze(-1).squeeze(-1)  # (B, 960)
        diff = torch.abs(f1 - f2)

        combined = torch.cat([f1, f2, diff], dim=1)  # (B, 2048*3)
        return self.mlp(combined).squeeze(1)         # (B,)

if __name__ == '__main__':
    model = IrisMatchModel()
    print(model)
