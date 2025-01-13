from torch import nn

class EmotionDetectModule(nn.Module):
    def __init__(self):
        super(EmotionDetectModule, self).__init__()
        out_neurons = 3
        self.emotion_ouput_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_neurons),
        )

    def forward(self, x_emotion):
        x_emotion = self.emotion_ouput_layer(x_emotion)
        return x_emotion