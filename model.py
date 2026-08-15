
import torch
import torch.nn as nn


# ============================================================
# BASIC RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.norm1 = nn.GroupNorm(
            8,
            channels
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.norm2 = nn.GroupNorm(
            8,
            channels
        )

        self.activation = nn.GELU()


    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual

        x = self.activation(x)

        return x


# ============================================================
# CHANNEL ATTENTION
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels):

        super().__init__()

        reduced = max(
            channels // 8,
            4
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Conv2d(
                channels,
                reduced,
                kernel_size=1
            ),

            nn.GELU(),

            nn.Conv2d(
                reduced,
                channels,
                kernel_size=1
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        attention = self.pool(x)

        attention = self.fc(attention)

        return x * attention


# ============================================================
# RESIDUAL + ATTENTION BLOCK
# ============================================================

class ResidualAttentionBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.residual = ResidualBlock(
            channels
        )

        self.attention = ChannelAttention(
            channels
        )


    def forward(self, x):

        x = self.residual(x)

        x = self.attention(x)

        return x


# ============================================================
# DOWN BLOCK
# ============================================================

class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.GroupNorm(
                8,
                out_channels
            ),

            nn.GELU(),

            ResidualAttentionBlock(
                out_channels
            )
        )


    def forward(self, x):

        return self.block(x)


# ============================================================
# UP BLOCK
# ============================================================

class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.fuse = nn.Sequential(

            nn.Conv2d(
                out_channels * 2,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.GroupNorm(
                8,
                out_channels
            ),

            nn.GELU(),

            ResidualAttentionBlock(
                out_channels
            )
        )


    def forward(
        self,
        x,
        skip
    ):

        x = self.up(x)

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.fuse(x)

        return x


# ============================================================
# KLAResUNet
# ============================================================

class KLAResUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        self.input_conv = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.GELU()
        )


        # ----------------------------------------------------
        # Encoder level 1
        # ----------------------------------------------------

        self.enc1 = nn.Sequential(

            ResidualAttentionBlock(32),

            ResidualAttentionBlock(32)
        )


        self.down1 = DownBlock(
            32,
            64
        )


        # ----------------------------------------------------
        # Encoder level 2
        # ----------------------------------------------------

        self.enc2 = nn.Sequential(

            ResidualAttentionBlock(64),

            ResidualAttentionBlock(64)
        )


        self.down2 = DownBlock(
            64,
            128
        )


        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        self.bottleneck = nn.Sequential(

            ResidualAttentionBlock(128),

            ResidualAttentionBlock(128),

            ResidualAttentionBlock(128)
        )


        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up2 = UpBlock(
            128,
            64
        )

        self.up1 = UpBlock(
            64,
            32
        )


        # ----------------------------------------------------
        # Super-resolution reconstruction
        # ----------------------------------------------------

        self.pre_sr = nn.Sequential(

            nn.Conv2d(
                32,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.GELU()
        )


        self.pixel_shuffle = nn.PixelShuffle(
            2
        )


        self.output_conv = nn.Conv2d(
            32,
            1,
            kernel_size=3,
            padding=1
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        x0 = self.input_conv(x)

        e1 = self.enc1(x0)

        d1 = self.down1(e1)

        e2 = self.enc2(d1)

        d2 = self.down2(e2)

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        b = self.bottleneck(d2)

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        u2 = self.up2(
            b,
            e2
        )

        u1 = self.up1(
            u2,
            e1
        )

        # ----------------------------------------------------
        # Super-resolution
        # ----------------------------------------------------

        x = self.pre_sr(u1)

        x = self.pixel_shuffle(x)

        x = self.output_conv(x)

        # ----------------------------------------------------
        # Output constrained to [0,1]
        # ----------------------------------------------------

        x = torch.sigmoid(x)

        return x
