import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, input_channel=1):
        super().__init__()
        self.down_sample = nn.MaxPool2d(2)
        self.down1 = ConvBlock(input_channel, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)
        self.down5 = ConvBlock(512, 1024)

    def forward(self, x):
        d1 = self.down1(x)                         # (B, 64,  H,    W)
        d2 = self.down2(self.down_sample(d1))      # (B, 128, H/2,  W/2)
        d3 = self.down3(self.down_sample(d2))      # (B, 256, H/4,  W/4)
        d4 = self.down4(self.down_sample(d3))      # (B, 512, H/8,  W/8)
        bottleneck = self.down5(self.down_sample(d4))  # (B, 1024, H/16, W/16)

        skips = [d1, d2, d3, d4]
        return skips, bottleneck


class Decoder(nn.Module):
    """
    输入:
      skips: [d1, d2, d3, d4]
      bottleneck
    输出:
      (B,1,H,W)
    """
    def __init__(self):
        super().__init__()
        self.up_sample5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up4 = ConvBlock(1024, 512)

        self.up_sample4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up3 = ConvBlock(512, 256)

        self.up_sample3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up2 = ConvBlock(256, 128)

        self.up_sample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.ReLU()
        )
        self.up1 = ConvBlock(128, 64)

        self.last = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, skips, bottleneck):
        d1, d2, d3, d4 = skips

        out = self.up4(torch.cat((self.up_sample5(bottleneck), d4), dim=1))
        out = self.up3(torch.cat((self.up_sample4(out), d3), dim=1))
        out = self.up2(torch.cat((self.up_sample3(out), d2), dim=1))
        out = self.up1(torch.cat((self.up_sample2(out), d1), dim=1))
        return self.last(out)


class DecoupleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(input_channel=1)
        self.decoder_vi = Decoder()
        self.decoder_ir = Decoder()
    def forward(self, f):
        skips_f,  bott_f  = self.encoder(f)
        out_vi = self.decoder_vi(skips_f, bott_f)
        out_ir = self.decoder_ir(skips_f, bott_f)
        return out_vi, out_ir