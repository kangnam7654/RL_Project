import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, num_action=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, 7, 3, 1, bias=False),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 7, 3, 1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 7, 3, 1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 7, 3, 1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)

        self.fc1 = nn.Sequential(
            nn.Linear(256, 512), nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc2 = nn.Linear(512, num_action)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                # nn.init.uniform_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
