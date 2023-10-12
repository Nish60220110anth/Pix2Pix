import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, asencoder: bool, use_dropout=False, act="relu", dropout=0.4):  # 0.5 is paper
        super(CNNBlock, self).__init__()

        if asencoder:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=4, stride=2, padding=1, padding_mode="reflect")
        else:  # decoder (4,2,1)
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(num_features=out_channels, affine=True),
                nn.ReLU() if act == "relu" else nn.LeakyReLU(negative_slope=0.2)
            )

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout2d(p=dropout)  # dropout 2d for cnn

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels, start_feature=64):
        super(Generator, self).__init__()

        self.initial_enc = nn.Sequential(
            nn.Conv2d(in_channels, start_feature, 4,
                      2, 1, padding_mode="reflect", bias=False),
            nn.LeakyReLU(negative_slope=0.2)
        )  # 128

        # LeakyRelu in encoder, RelU in decoder

        self.encoder1 = CNNBlock(start_feature, start_feature*2, asencoder=True, use_dropout=False,
                                 act="lrelu")  # 64
        self.encoder2 = CNNBlock(start_feature*2, start_feature*4, asencoder=True, use_dropout=False,
                                 act="lrelu")  # 32
        self.encoder3 = CNNBlock(start_feature*4, start_feature*8, asencoder=True, use_dropout=False,
                                 act="lrelu")  # 16
        self.encoder4 = CNNBlock(start_feature*8, start_feature*8, asencoder=True, use_dropout=False,
                                 act="lrelu")  # 8
        self.encoder5 = CNNBlock(start_feature*8, start_feature*8, asencoder=True, use_dropout=False,
                                 act="lrelu")  # 4
        self.encoder6 = CNNBlock(start_feature*8, start_feature*8, asencoder=True, use_dropout=False,
                                 act="lrelu")  # 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=start_feature*8, out_channels=start_feature*8, stride=2, kernel_size=4, padding=1,
                      padding_mode="reflect", bias=True),
            nn.ReLU()
        )  # start*8 x 1 x 1

        # only first 3 decoders must use dropouts
        self.decoder1 = CNNBlock(start_feature*8, start_feature*8, asencoder=False, use_dropout=True,
                                 act="relu", dropout=0.4)  # skip connection x 2
        self.decoder2 = CNNBlock(start_feature*8*2, start_feature*8, asencoder=False, use_dropout=True,
                                 act="relu", dropout=0.4)
        self.decoder3 = CNNBlock(start_feature*8*2, start_feature*8, asencoder=False, use_dropout=True,
                                 act="relu", dropout=0.4)
        self.decoder4 = CNNBlock(start_feature*8*2, start_feature*8, asencoder=False, use_dropout=False,
                                 act="relu")
        self.decoder5 = CNNBlock(start_feature*8*2, start_feature*4, asencoder=False, use_dropout=False,
                                 act="relu")
        self.decoder6 = CNNBlock(start_feature*4*2, start_feature*2, asencoder=False, use_dropout=False,
                                 act="relu")
        self.decoder7 = CNNBlock(start_feature*2*2, start_feature, asencoder=False, use_dropout=False,
                                 act="relu")
        self.final_decoder = nn.Sequential(
            nn.ConvTranspose2d(start_feature*2, in_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        en1 = self.initial_enc(x)
        en2 = self.encoder1(en1)
        en3 = self.encoder2(en2)
        en4 = self.encoder3(en3)
        en5 = self.encoder4(en4)
        en6 = self.encoder5(en5)
        en7 = self.encoder6(en6)

        bottleout = self.bottleneck(en7)

        de1 = self.decoder1(bottleout)
        de2 = self.decoder2(torch.cat([de1, en7], dim=1))
        de3 = self.decoder3(torch.cat([de2, en6], dim=1))
        de4 = self.decoder4(torch.cat([de3, en5], dim=1))
        de5 = self.decoder5(torch.cat([de4, en4], dim=1))
        de6 = self.decoder6(torch.cat([de5, en3], dim=1))
        de7 = self.decoder7(torch.cat([de6, en2], dim=1))

        return self.final_decoder(torch.cat([de7, en1], dim=1))
