from torch import nn

class PoseDetectModule(nn.Module):
    def __init__(self):
        super(PoseDetectModule, self).__init__()
        out_neurons = 3
        self.pose_ouput_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_neurons),
        )

    def forward(self, x_pose):
        x_pose = self.pose_ouput_layer(x_pose)
        return x_pose