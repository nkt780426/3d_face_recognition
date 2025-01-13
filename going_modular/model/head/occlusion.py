import torch
from torch import nn

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class OcclusionDetectModule(nn.Module):
    def __init__(self):
        super(OcclusionDetectModule, self).__init__()
        out_neurons = 3
        self.occlusion_ouput_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, out_neurons),
        )

    def forward(self, x_occlusion):
        x_occlusion = self.occlusion_ouput_layer(x_occlusion)
        return x_occlusion