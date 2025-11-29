from torch import nn


class AutoDriveNet(nn.Module):
    def __init__(self):
        super(AutoDriveNet, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
            nn.SELU(),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.SELU(),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2),
            nn.SELU(),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3),
            nn.SELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.Dropout(0.5)
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=64 * 8 * 13, out_features=100),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=100, out_features=50),
            nn.SELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.SELU(),
            nn.Linear(in_features=10, out_features=10),
            nn.SELU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=10, out_features=10),
            nn.SELU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=10, out_features=10),
            nn.SELU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), 3, 120, 160)
        x = self.convolution(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x