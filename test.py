####################################################
# This script uses Generator network to generate High Resolution (4x) images from the Low resolution images
####################################################

import torch
from torchvision import transforms,datasets
from src import model,loss
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import argparse
import imageio
import os

torch.set_default_tensor_type(torch.cuda.FloatTensor)

def test(args):
    generator = model.Generator().cuda()

    gen_model = args.gen_model
    input_folder = args.input_folder
    output_folder = args.input_folder

    generator.load_state_dict(torch.load(gen_model))

    test_files = os.listdir(input_folder)

    for test_file in test_files:
        img_path = input_folder+'/'+test_file
        inp = imageio.imread(img_path)
        if len(inp.shape) == 3:
            out = generator(torch.from_numpy((inp/255).transpose((2,0,1))).cuda().type(torch.cuda.FloatTensor).unsqueeze(0))
        else:
            inp = np.array([inp,inp,inp])
            print(inp.shape)
            out = generator(torch.from_numpy(inp/255).cuda().type(torch.cuda.FloatTensor).unsqueeze(0))
        out = out.clamp(min=-1,max=1)
        out = (out+1)/2
        plt.figure()
        plt.imshow(out[0].cpu().detach().numpy().transpose(1,2,0))
        plt.imsave(output_folder+'/output_' + test_file,out[0].cpu().detach().numpy().transpose(1,2,0))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model",type=str,default=None,help="Path to weights of Generator")
    parser.add_argument("--input_folder",type=str,default=None,help="Path to input image folder")
    parser.add_argument("--output_folder",type=str,default=None,help="Path to output image folder")
    args = parser.parse_args()
    test(args)
