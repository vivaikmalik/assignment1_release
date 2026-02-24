import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride_dw, stride_pw):
        super().__init__()
        """
        Build the depthwise separable convolution layer
        For the depthwise convolution (use padding=1 and bias=False for the convolution)
        For the pointwise convolution (use padding=0 and bias=False fot the convolution)

        Inputs:
            in_channels: number of input channels
            out_channels: number of output channels
            stride_dw: stride for depthwise convolution
            stride_pw: stride for pointwise convolution
        """
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride_dw, groups=in_channels, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_pw, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        Build the MobileNet architecture
        For the first standard convolutional layer (use padding=1 and bias=False for the convolution)
        For the AvgPool layer, use nn.AdaptiveAvgPool2d.

        Inputs:
            num_classes: number of classes for classification
        """
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dw_sep_conv0 = DepthwiseSeparableConv(in_channels=32, out_channels=64, stride_dw=1, stride_pw=1)
        self.dw_sep_conv1 = DepthwiseSeparableConv(in_channels=64, out_channels=128, stride_dw=2, stride_pw=1)
        self.dw_sep_conv2 = DepthwiseSeparableConv(in_channels=128, out_channels=128, stride_dw=1, stride_pw=1)
        self.dw_sep_conv3 = DepthwiseSeparableConv(in_channels=128, out_channels=256, stride_dw=2, stride_pw=1)
        self.dw_sep_conv4 = DepthwiseSeparableConv(in_channels=256, out_channels=256, stride_dw=1, stride_pw=1)
        self.dw_sep_conv5 = DepthwiseSeparableConv(in_channels=256, out_channels=512, stride_dw=2, stride_pw=1)
        self.dw_sep_conv61 = DepthwiseSeparableConv(in_channels=512, out_channels=512, stride_dw=1, stride_pw=1)
        self.dw_sep_conv62 = DepthwiseSeparableConv(in_channels=512, out_channels=512, stride_dw=1, stride_pw=1)
        self.dw_sep_conv63 = DepthwiseSeparableConv(in_channels=512, out_channels=512, stride_dw=1, stride_pw=1)
        self.dw_sep_conv64 = DepthwiseSeparableConv(in_channels=512, out_channels=512, stride_dw=1, stride_pw=1)
        self.dw_sep_conv65 = DepthwiseSeparableConv(in_channels=512, out_channels=512, stride_dw=1, stride_pw=1)
        self.dw_sep_conv7 = DepthwiseSeparableConv(in_channels=512, out_channels=1024, stride_dw=2, stride_pw=1)
        self.dw_sep_conv8 = DepthwiseSeparableConv(in_channels=1024, out_channels=1024, stride_dw=2, stride_pw=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.dw_sep_conv0(x)
        x = self.dw_sep_conv1(x)
        x = self.dw_sep_conv2(x)
        x = self.dw_sep_conv3(x)
        x = self.dw_sep_conv4(x)
        x = self.dw_sep_conv5(x)
        x = self.dw_sep_conv61(x)
        x = self.dw_sep_conv62(x)
        x = self.dw_sep_conv63(x)
        x = self.dw_sep_conv64(x)
        x = self.dw_sep_conv65(x)
        x = self.dw_sep_conv7(x)
        x = self.dw_sep_conv8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x