import torch
import torchsummary
import numpy as np
from torch import nn
from torchvision.models import vgg19

torch.set_default_tensor_type(torch.cuda.FloatTensor)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss,self).__init__()
        self.mse_loss = nn.MSELoss()
        self.vgg = list(vgg19(pretrained=True).children())[0]
        for param in self.vgg.parameters():
            param.requires_grad = False
    def forward(self,generated_image,ground_truth):
        y = ground_truth
        y_hat = generated_image
        loss = 0
        for name,module in self.vgg._modules.items():
            if isinstance(module,nn.MaxPool2d):
                loss = loss + self.mse_loss(y_hat/12.75,y.detach()/12.75)
            y = module(y)
            y_hat = module(y_hat)
        return loss

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss,self).__init__()
    def forward(self,probabilities):
        return -torch.log(probabilities).sum()

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss,self).__init__()
    def forward(self,probabilities,is_ground_truth):
        if is_ground_truth:
            return -torch.log(probabilities).sum()
        else:
            return torch.log(probabilities).sum()
