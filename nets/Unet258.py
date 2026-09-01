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


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        # query/key/value: (B, L, C) -> (L, B, C)
        query = query.transpose(0, 1)
        key   = key.transpose(0, 1)
        value = value.transpose(0, 1)

        attn_output, _ = self.multihead_attn(query, key, value)

        # (L, B, C) -> (B, L, C)
        return attn_output.transpose(0, 1)


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


class FRMNet(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8):
        super().__init__()
        self.encoder_f_vi = Encoder(input_channel=1)
        self.encoder_f_ir = Encoder(input_channel=1)
        # self.encoder_vi = Encoder(input_channel=1)
        # self.encoder_ir = Encoder(input_channel=1)
        self.encoder = Encoder(input_channel=1)
        self.decoder_vi = Decoder()
        self.decoder_ir = Decoder()

        self.ca_vi = CrossAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ca_ir = CrossAttention(embed_dim=embed_dim, num_heads=num_heads)

    def _feat_to_seq(self, x):
        # x: (B, C, H, W) -> (B, L, C)
        B, C, H, W = x.shape
        return x.view(B, C, H * W).transpose(1, 2), (H, W)

    def _seq_to_feat(self, seq, spatial):
        # seq: (B, L, C) -> (B, C, H, W)
        B, L, C = seq.shape
        H, W = spatial
        return seq.transpose(1, 2).view(B, C, H, W)

    def forward(self, vi, ir, f):
        skips_f_vi,  bott_f_vi  = self.encoder_f_vi(f)
        skips_f_ir,  bott_f_ir  = self.encoder_f_ir(f)

        skips_vi, bott_vi = self.encoder_f_vi(vi)
        skips_ir, bott_ir = self.encoder_f_ir(ir)

        seq_f_vi,  spatial = self._feat_to_seq(bott_f_vi)
        seq_f_ir,  spatial = self._feat_to_seq(bott_f_ir)
        seq_vi, _       = self._feat_to_seq(bott_vi)
        seq_ir, _       = self._feat_to_seq(bott_ir)

        seq_f_vi = self.ca_vi(query=seq_f_vi, key=seq_vi, value=seq_vi)
        seq_f_ir = self.ca_ir(query=seq_f_ir, key=seq_ir, value=seq_ir)

        bott_f_vi = self._seq_to_feat(seq_f_vi, spatial)
        bott_f_ir = self._seq_to_feat(seq_f_ir, spatial)

        out_vi = self.decoder_vi(skips_vi, bott_f_vi)
        out_ir = self.decoder_ir(skips_ir, bott_f_ir)




        # skips_f,  bott_f  = self.encoder(f)
        # skips_vi, bott_vi = self.encoder(vi)
        # skips_ir, bott_ir = self.encoder(ir)

        # seq_f,  spatial = self._feat_to_seq(bott_f)
        # seq_vi, _       = self._feat_to_seq(bott_vi)
        # seq_ir, _       = self._feat_to_seq(bott_ir)

        # seq_f_vi = self.ca_vi(query=seq_f, key=seq_vi, value=seq_vi)
        # seq_f_ir = self.ca_ir(query=seq_f, key=seq_ir, value=seq_ir)

        # bott_f_vi = self._seq_to_feat(seq_f_vi, spatial)
        # bott_f_ir = self._seq_to_feat(seq_f_ir, spatial)

        # out_vi = self.decoder_vi(skips_vi, bott_f_vi)
        # out_ir = self.decoder_ir(skips_ir, bott_f_ir)
        return out_vi, out_ir