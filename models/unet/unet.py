from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, inc_channels=16, **kwargs):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, inc_channels)

        self.down1 = Down(inc_channels, inc_channels * 2)
        self.down2 = Down(inc_channels * 2, inc_channels * 4)
        self.down3 = Down(inc_channels * 4, inc_channels * 8)
        self.down4 = Down(inc_channels * 8, inc_channels * 8)

        self.up4 = Up(inc_channels * 16, inc_channels * 4)
        self.up3 = Up(inc_channels * 8, inc_channels * 2)
        self.up2 = Up(inc_channels * 4, inc_channels)
        self.up1 = Up(inc_channels * 2, inc_channels)

        self.outc = OutConv(inc_channels, n_classes)

    def forward(self, x):
        d0 = self.inc(x)  # 32

        d1 = self.down1(d0)  # 64
        d2 = self.down2(d1)  # 128
        d3 = self.down3(d2)  # 256
        d4 = self.down4(d3)  # 256

        u4 = self.up4(d4, d3)  # 128
        u3 = self.up3(u4, d2)  # 64
        u2 = self.up2(u3, d1)  # 32
        u1 = self.up1(u2, d0)  # 32

        u0 = self.outc(u1)  # 3
        out = torch.sigmoid(u0)

        # nchw -> nhwc
        out = out.permute(0, 2, 3, 1)
        return out
