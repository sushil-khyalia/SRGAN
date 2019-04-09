import torch
import torchsummary
import numpy as np
from torch import nn
from helperLayers import *

torch.set_default_tensor_type(torch.cuda.FloatTensor)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,kernel_size=9,out_channels=64,padding=4,stride=1)
        self.prelu1 = nn.PReLU()
        self.resblock1 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock2 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock3 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock4 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock5 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock6 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock7 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock8 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock9 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock10 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock11 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock12 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock13 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock14 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock15 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.resblock16 = ResBlock(channels=64,kernel_size=3,padding=1,stride=1)
        self.conv2 = nn.Conv2d(in_channels=64,kernel_size=3,out_channels=64,padding=1,stride=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        self.subpixelconv1 = SubPixelConv(in_channels=64,kernel_size=3,out_channels=256,padding=1,stride=1)
        self.subpixelconv2 = SubPixelConv(in_channels=64,kernel_size=3,out_channels=256,padding=1,stride=1)
        self.conv3 = nn.Conv2d(in_channels=64,kernel_size=9,out_channels=3,padding=4,stride=1)

    def forward(self,inp):
        x = self.conv1(inp)
        x = self.prelu1(x)
        y = self.resblock1(x)
        y = self.resblock2(y)
        y = self.resblock3(y)
        y = self.resblock4(y)
        y = self.resblock5(y)
        y = self.resblock6(y)
        y = self.resblock7(y)
        y = self.resblock8(y)
        y = self.resblock9(y)
        y = self.resblock10(y)
        y = self.resblock11(y)
        y = self.resblock12(y)
        y = self.resblock13(y)
        y = self.resblock14(y)
        y = self.resblock15(y)
        y = self.resblock16(y)
        y = self.conv2(y)
        y = self.batch_norm1(y)
        y = self.subpixelconv1(x+y)
        y = self.subpixelconv2(y)
        output = self.conv3(y)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,kernel_size=3,out_channels=64,padding=1,stride=1)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.convblock1 = ConvBlock(kernel_size=3,in_channels=64,out_channels=64,padding=1,stride=2)
        self.convblock2 = ConvBlock(kernel_size=3,in_channels=64,out_channels=128,padding=1,stride=1)
        self.convblock3 = ConvBlock(kernel_size=3,in_channels=128,out_channels=128,padding=1,stride=2)
        self.convblock4 = ConvBlock(kernel_size=3,in_channels=128,out_channels=256,padding=1,stride=1)
        self.convblock5 = ConvBlock(kernel_size=3,in_channels=256,out_channels=256,padding=1,stride=2)
        self.convblock6 = ConvBlock(kernel_size=3,in_channels=256,out_channels=512,padding=1,stride=1)
        self.convblock7 = ConvBlock(kernel_size=3,in_channels=512,out_channels=512,padding=1,stride=2)
        self.dense1 = nn.Linear(in_features=18432,out_features=1024)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.dense2 = nn.Linear(in_features=1024,out_features=1)
        self.sigmoid1 = nn.Sigmoid()
    def forward(self,inp):
        x = self.conv1(inp)
        x = self.leaky_relu1(x)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = x.view(x.size(0),-1)
        x = self.dense1(x)
        x = self.leaky_relu2(x)
        x = self.dense2(x)
        output = self.sigmoid1(x)
        return output
