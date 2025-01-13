from torch import nn

class OcclusionDetectModule(nn.Module):
    def __init__(self):
        super(OcclusionDetectModule, self).__init__()
        out_neurons = 3
        self.occlusion_ouput_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_neurons),
        )

    def forward(self, x_occlusion):
        x_occlusion = self.occlusion_ouput_layer(x_occlusion)
        return x_occlusion