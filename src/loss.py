import torch
import torchsummary
import numpy as np
from torch import nn
from torchvision.models import vgg19

torch.set_default_tensor_type(torch.cuda.FloatTensor)


# class defining perceptual loss using feature maps of pretrained VGG19 network
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss,self).__init__()
        self.mse_loss = nn.MSELoss()
        self.vgg = list(vgg19(pretrained=True).children())[0] #loading pretrained VGG19 network
        for param in self.vgg.parameters():
            param.requires_grad = False #freezing weights of VGG network
    def forward(self,generated_image,ground_truth):
        y = ground_truth
        y_hat = generated_image
        loss = 0
        for name,module in self.vgg._modules.items():
            if isinstance(module,nn.MaxPool2d):
                # adding mean square loss on each convolutional layer just before the max pooling layer
                loss = loss + self.mse_loss(y_hat/12.75,y.detach()/12.75)
            y = module(y)
            y_hat = module(y_hat)
        return loss
