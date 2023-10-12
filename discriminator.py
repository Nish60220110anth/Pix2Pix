import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, **kwargs):
        super(CNNBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride,
                      padding_mode="reflect", bias=False, kernel_size=4, **kwargs, padding=1),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels, num_features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels, out_channels=num_features[0], kernel_size=4,
                      stride=2, padding=1, padding_mode="reflect", bias=True),
            nn.LeakyReLU(negative_slope=0.2)
        )

        cnn_layers = []
        in_channel = num_features[0]

        for feature in num_features[1:]:
            cnn_layers.append(
                CNNBlock(in_channels=in_channel, out_channels=feature,
                         stride=1 if feature == num_features[-1] else 2))
            in_channel = feature

        self.cnnblocks = nn.Sequential(*cnn_layers,
                                       nn.Conv2d(in_channels=in_channel, out_channels=1, stride=1, padding_mode="reflect", kernel_size=4, padding=1))

    def forward(self, x, y):  # x gen, y real
        # along channels [6 3+3]
        return self.cnnblocks(self.initial(torch.cat([x, y], dim=1)))
