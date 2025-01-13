from torch import nn

class GenderDetectModule(nn.Module):
    def __init__(self):
        super(GenderDetectModule, self).__init__()
        out_neurons = 2
        self.gender_ouput_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_neurons),
        )

    def forward(self, x_gender):
        x_gender = self.gender_ouput_layer(x_gender)
        return x_gender