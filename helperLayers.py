import torch
import torchsummary
import numpy as np
from torch import nn

torch.set_default_tensor_type(torch.cuda.DoubleTensor)

class ResBlock(nn.Module):
    def __init__(self,kernel_size,channels,padding,stride):
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=kernel_size,padding=padding,stride=stride)
        self.batch_norm1 = nn.BatchNorm2d(num_features=channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=kernel_size,padding=padding,stride=stride)
        self.batch_norm2 = nn.BatchNorm2d(num_features=channels)
    def forward(self,inp):
        x = self.conv1(inp)
        x = self.batch_norm1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        output = x + inp
        return output

class SubPixelConv(nn.Module):
    def __init__(self,kernel_size,in_channels,out_channels,padding,stride):
        super(SubPixelConv,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,stride=stride)
        self.pixel_shuffler1 = nn.PixelShuffle(2)
        self.prelu1 = nn.PReLU()
    def forward(self,inp):
        x = self.conv1(inp)
        x = self.pixel_shuffler1(x)
        output = self.prelu1(x)
        return output

class ConvBlock(nn.Module):
    def __init__(self,kernel_size,in_channels,out_channels,padding,stride):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,stride=stride)
        self.batch_norm1 = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)
    def forward(self,inp):
        x = self.conv1(inp)
        x = self.batch_norm1(x)
        output = self.leaky_relu1(x)
        return output
