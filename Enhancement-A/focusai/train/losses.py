import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import torchvision.models as models
import kornia

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=[4, 9, 16], style_weight=0.0):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        start = 0
        for end in layers:
            block = nn.Sequential(*[vgg[i] for i in range(start, end)])
            for p in block.parameters():
                p.requires_grad = False
            self.slices.append(block)
            start = end

        self.style_weight = style_weight
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, h * w)
        return torch.matmul(f, f.transpose(1, 2)) / (c * h * w)

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0

        for block in self.slices:
            pred = block(pred)
            target = block(target)
            loss += F.l1_loss(pred, target)
            if self.style_weight > 0:
                loss += self.style_weight * F.l1_loss(self.gram(pred), self.gram(target))
        return loss

def color_loss(pred, target):
    pred_lab = kornia.color.rgb_to_lab(pred)
    target_lab = kornia.color.rgb_to_lab(target)
    return F.l1_loss(pred_lab, target_lab)

def total_loss(pred, target, vgg_loss_fn):
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    l1 = F.l1_loss(pred, target)
    ssim_val = 1 - ssim(pred, target, data_range=1.0, size_average=True)
    percept = vgg_loss_fn(pred, target)
    col_loss = color_loss(pred, target)

    loss = (
        0.3 * l1 +
        0.3 * ssim_val +
        0.3 * percept +
        0.1 * col_loss
    )

    loss_dict = {
        "l1": l1.item(),
        "ssim": (1 - ssim_val).item(),
        "percept": percept.item(),
        "color": col_loss.item(),
        "total": loss.item(),
    }

    return loss, loss_dict
