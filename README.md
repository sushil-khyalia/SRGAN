# SRGAN

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Networks

## Abstract

This Project is the pytorch implementation of the SRGAN (Super Resolution Generative Adverserial Network) based on the 2017 CVPR paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802.pdf) which presented state-of-the-art results for Single Image Super Resolution (SISR). The following model was used for the task of Image SR:

![alt text](https://github.com/sushil-khyalia/SRGAN/blob/master/img/model.jpeg "SRGAN Model")

Instead of using the pixel wise loss function like MSE, commonly used for this task, the authors proposed a perceptual loss function which is a combination of content-loss (based on high level feature maps of VGG Network) and adverserial loss, which produced SR images much closer to the Natural Image manifold as compared to overly smooth textures produced using MSE based loss function. The following figures shows this phenomenon

![alt test](https://github.com/sushil-khyalia/SRGAN/blob/master/img/srgan_bicubic.PNG "srgan_vs_bicubic")

## Requirements

Kindly use the requirements.txt to set up your machine for replicating this experiment. some dependendecies are :

* pytorch: 1.1.0
* torchvision: 0.2.2
* numpy: 1.16.3
* scipy: 1.2.1
* matplotlib: 3.0.3
* imageio 2.4.1
* skimage 0.14.2

## Datasets

### Train Dataset

The network was trained using images from __ILSVRC2014 DET__ Dataset containing a set 456,567 training images. We performed random crops of size (96,96) and random horizontal flips on each image.

### Test Dataset

Test Images have been sampled from the widely used bechmark datasets __SET5__ [ Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html) | __SET14__ [ Zeyde et al. LNCS 2010 ](https://sites.google.com/site/romanzeyde/research-interests) | __BSD100__ [Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/). They can be downloaded from [here](https://drive.google.com/file/d/1kuDs1BgkY12ztUVogDmFNEjvKobCyuuQ/view?usp=sharing).

## Usage

### Train

> python train.py --data_path <Path to the training dataset>
>                 --batch_size <Batch Size to be used for training>
>                
>optional Arguments:
>--gen_model               <Path to pretrained-weights of Generator>, Default: Random Initialisation
>--disc_model              <Path to pretrained-weights of Discriminator>, Default: Random Initialisation
>--num_epochs              <number of epochs to train>, Default 30
>--train_discriminator     <Set true to train both Generator and Discriminator>, Default=False

### Test

>python test.py --gen_model <Path to weights of Generator>
>                --input_folder <Path to input image folder>
>                --output Folder <Path to output image folder>`

## Results

A comparison of PSNR(Peak Signal To Noise Ration) and SSIM (Structural Similarity Index) for Nearest Neighbours, cubic interpolation, our SRGAN model implementation and author's implementation taken from [here](https://github.com/tensorlayer/srgan/releases/tag/1.2.0). We used the skimage library to compute [psnr](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_psnr) and [ssim](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim) on the y-channel of the image (converted to ycbcr format).

#### Results on SET5

| SET5 | nearest | bicubic | CVPR17 | Ours |
| ---- |:-------:| :-----: | :----: | :--: |
| PSNR | 26.2582 | 28.4304 | 28.2102 | 27.2761 |
| SSIM | 0.7639 | 0.8231 | 0.8323 | 0.8162 |

#### Results on SET14

| SET5 | nearest | bicubic | CVPR17 | Ours |
| ---- |:-------:| :-----: | :----: | :--: |
| PSNR | 24.8309 | 26.2132 | 26.1595 | 25.6378 |
| SSIM | 0.6909 | 0.7312 | 0.7329 | 0.7314 |

#### Results on BSD 100

| SET5 | nearest | bicubic | CVPR17 | Ours |
| ---- |:-------:| :-----: | :----: | :--: |
| PSNR | 25.037 | 25.9613 | 24.6297 | 24.5921 |
| SSIM | 0.6539 | 0.6856 | 0.6643 | 0.6637 |

Shown below is a comparison of 3 different upsclaing methods:

![alt test](https://github.com/sushil-khyalia/SRGAN/blob/master/img/img1.PNG "comparison of different methods")

Here are some more results of SR of Images from from __SET5__, __SET14__ and __BSD100__  using Binlear Interpolation(left), our SRGAN model (middle) and ground Truth(right):

![alt test](https://github.com/sushil-khyalia/SRGAN/blob/master/img/img2.PNG "img1")

![alt test](https://github.com/sushil-khyalia/SRGAN/blob/master/img/img3.PNG "img2")

![alt test](https://github.com/sushil-khyalia/SRGAN/blob/master/img/img4.PNG "img3")

![alt test](https://github.com/sushil-khyalia/SRGAN/blob/master/img/img5.PNG "img4")

![alt test](https://github.com/sushil-khyalia/SRGAN/blob/master/img/img6.PNG "img5")



                


