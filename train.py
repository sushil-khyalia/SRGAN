import torch
from torchvision import transforms,datasets
from src import model,loss
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import argparse

torch.set_default_tensor_type(torch.cuda.FloatTensor)

def train(args):
    data_path = args.data_path              # argument to get data path
    train_dataset = datasets.ImageFolder(      # load data from the given path
    data_path,
    transforms.Compose([
        transforms.RandomResizedCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))

    batch_size = args.batch_size            # get batch size

    data_loader = torch.utils.data.DataLoader(train_dataset,        # create data loader object to iterate over data
                                          batch_size=batch_size,
                                          shuffle=True)

    # initialize generator, discriminator and loss network
    generator = model.Generator().cuda()
    discriminator = model.Discriminator().cuda()
    loss_network = loss.PerceptualLoss().cuda()

    # initialize optimizers
    optim_gen = torch.optim.Adam(lr=1e-4,params=generator.parameters())
    optim_disc = torch.optim.Adam(lr=1e-4,params=discriminator.parameters())

    gen_model = args.gen_model

    if gen_model is not None:
        generator.load_state_dict(torch.load(gen_model)) # load a pretrained generator network

    disc_model = args.disc_model

    if disc_model is not None:
        discriminator.load_state_dict(torch.load(disc_model)) # load a pretrained discriminator network

    num_epochs = args.num_epochs
    train_discriminator = args.train_discriminator
    for epoch in range(num_epochs):
        sum_loss_generator = 0
        sum_loss_between_generator = 0
        sum_loss_discriminator = 0
        sum_loss_between_discriminator = 0
        print('=========================================================')
        for step,data in enumerate(data_loader):
            hr_images = data[0].cuda()
            lr_images = np.zeros((hr_images.shape[0],hr_images.shape[1],int(hr_images.shape[2]/4),int(hr_images.shape[3]/4)))
            # generate low resolution images from the high dimesional images
            for i in range(hr_images.shape[0]):
                lr_image = scipy.misc.imresize(hr_images[i].cpu().numpy(),size=0.25,interp='bicubic')/255
                lr_images[i] = lr_image.transpose((2,0,1))
            lr_images = torch.from_numpy(lr_images)

            generated_imgs = generator(lr_images.cuda().type(torch.cuda.FloatTensor)) # get output from generator
            if train_discriminator:
                generated_probs = discriminator(generated_imgs)     # get output from discriminator
            ####################################################################################
            if train_discriminator:
                # compute loss of discriminator and perform one optimization update on discriminator
                discriminator.zero_grad()
                loss_images = (-discriminator(2*hr_images-1).log().sum() - (1-generated_probs).log().sum())/32
                loss_images.backward(retain_graph=True)
                optim_disc.step()

                sum_loss_discriminator = sum_loss_discriminator + loss_images.cpu().detach().numpy()
                sum_loss_between_discriminator = sum_loss_between_discriminator + loss_images.cpu().detach().numpy()
                if step%50 == 49:
                    print("Epoch Number : "+str(epoch+1)+" Step Number : "+str(step+1)+" Loss Discriminator: "+str(sum_loss_between_discriminator/50))
                    sum_loss_between_discriminator = 0
            ####################################################################################
            generator.zero_grad()
            if train_discriminator:
                loss_images = loss_network(generated_imgs,2*hr_images-1) - generated_probs.log().sum()*1e-3/16      # compute loss of generator as a combination of perceptual loss and adversarial loss
            else:
                loss_images = loss_network(generated_imgs,2*hr_images-1)                                            # compute loss of generator as perceptual loss only
            loss_images.backward(retain_graph=True)
            optim_gen.step() # perform one optimization update on generator

            sum_loss_generator = sum_loss_generator + loss_images.cpu().detach().numpy()
            sum_loss_between_generator = sum_loss_between_generator + loss_images.cpu().detach().numpy()
            if step%50 == 49:
                print("Epoch Number : "+str(epoch+1)+" Step Number : "+str(step+1)+" Loss Generator: "+str(sum_loss_between_generator/50))
                sum_loss_between_generator = 0
        print('==========================================================')
        print('Epoch '+str(epoch+1)+' completed. Loss Generator : '+str(sum_loss_generator/len(data_loader)))
        if train_discriminator:
            print('Epoch '+str(epoch+1)+' completed. Loss Discriminator : '+str(sum_loss_discriminator/len(data_loader)))
        file_name_gen = 'generator'+str(epoch+1)+'.weights'
        torch.save(generator.state_dict(),file_name_gen) # save generator weights
        if train_discriminator:
            file_name_disc = 'discriminator'+str(epoch+1)+'.weights'
            torch.save(discriminator.state_dict(),file_name_disc)   # save discriminator weights

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str,help="Path to the training dataset")
    parser.add_argument("--batch_size",type=int,help="Batch Size to be used for training")
    parser.add_argument("--gen_model",type=str,default=None,help="Path to weights of Generator")
    parser.add_argument("--disc_model",type=str,default=None,help="Path to weights of Discriminator")
    parser.add_argument("--num_epochs",type=int,help="Number of epochs to train")
    parser.add_argument("--train_discriminator",type=lambda x: (str(x).lower() in ['true','1', 'yes']),default=False,help="Set true to train both Generator and Discriminator")
    args = parser.parse_args()
    train(args)
