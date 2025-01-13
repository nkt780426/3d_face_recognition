from torch import nn

class SpectacleDetectModule(nn.Module):
    def __init__(self):
        super(SpectacleDetectModule, self).__init__()
        out_neurons = 2
        self.spectacle_ouput_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_neurons)
        )

    def forward(self, x_spectacle):
        x_spectacle = self.spectacle_ouput_layer(x_spectacle)
        return x_spectacle