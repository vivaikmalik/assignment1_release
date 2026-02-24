import torch
from torch import nn


def double_conv_block(in_channels, out_channels):
    """
    This double conv block are the blocks used in the encoder part of UNet.
    It uses a padding of 1 to preserve spatial dimensions.
    
    :param in_channels: Description
    :param out_channels: Description
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class DecoderBlock(nn.Module):
    """
    Decoder block of UNet. It consists of an upconvolution layer followed by a double conv block.
    Use the double_conv_block defined above.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = double_conv_block(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.encoder_block1 = double_conv_block(input_shape, 64)
        self.encoder_block2 = double_conv_block(64, 128)
        self.encoder_block3 = double_conv_block(128, 256)
        self.encoder_block4 = double_conv_block(256, 512)
        self.encoder_block5 = double_conv_block(512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder_block1 = DecoderBlock(1024, 512)
        self.decoder_block2 = DecoderBlock(512, 256)
        self.decoder_block3 = DecoderBlock(256, 128)
        self.decoder_block4 = DecoderBlock(128,64)
        self.outconv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        skip1 = self.encoder_block1(x)
        x = self.pool(skip1 )
        skip2 = self.encoder_block2(x)
        x = self.pool(skip2)
        skip3 = self.encoder_block3(x)
        x = self.pool(skip3)
        skip4 = self.encoder_block4(x)
        x = self.pool(skip4)
        x = self.encoder_block5(x)
        x = self.decoder_block1(x, skip4)
        x = self.decoder_block2(x, skip3)
        x = self.decoder_block3(x, skip2)
        x = self.decoder_block4(x, skip1)
        x = self.outconv(x)
        return x