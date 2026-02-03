import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch//r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//r, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1)
        )
        self.att = SEBlock(ch)

    def forward(self, x):
        return F.relu(x + self.att(self.conv(x)))

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(out_ch)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(out_ch)
        )

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([x, skip], dim=1))

class ResidualAttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = Down(1, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)

        self.bottleneck = ResBlock(256)

        self.u3 = Up(256, 256, 128)
        self.u2 = Up(128, 128, 64)
        self.u1 = Up(64, 64, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        s1, p1 = self.d1(x)
        s2, p2 = self.d2(p1)
        s3, p3 = self.d3(p2)

        b = self.bottleneck(p3)

        d3 = self.u3(b, s3)
        d2 = self.u2(d3, s2)
        d1 = self.u1(d2, s1)

        return self.final(d1) + x
