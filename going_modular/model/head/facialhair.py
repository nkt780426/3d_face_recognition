from torch import nn

class FacialHairDetectModule(nn.Module):
    def __init__(self):
        super(FacialHairDetectModule, self).__init__()
        out_neurons = 2
        self.facial_hair_ouput_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_neurons),
        )

    def forward(self, x_facial_hair):
        x_facial_hair = self.facial_hair_ouput_layer(x_facial_hair)
        return x_facial_hair