import argparse
from torchvision import transforms
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import helper
from glob import glob
import os
from dcgan_itself import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model_wxj/model_epoch_2.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=9, help='Number of generated outputs')
args = parser.parse_args()

# Load the checkpoint file.
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']


for i in range(500):
    # Create the generator network.
    netG = Generator(params).to(device)
    # Load the trained generator weights.
    netG.load_state_dict(state_dict['generator'])
    print(netG)

    print(args.num_output)
    # Get latent vector Z from unit normal distribution.
    noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)


    def output_fig(images_array, file_name="./results"):

        plt.figure(figsize=(6, 6), dpi=100)
        plt.imshow(helper.images_square_grid(images_array))
        plt.axis("off")
        plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)

    # Turn off gradient calculation to speed up the process.
    with torch.no_grad():
	    # Get generated image from the noise vector using
	    # the trained generator.
        # generated_img = netG(noise).detach().cpu()

        generated_img = netG(noise)
        generated_img = generated_img.permute(0,2,3,1)


    print(generated_img.shape)  # should be (9, width, height, 3)
    # width = height = 56
    output_fig(generated_img.cpu().detach().numpy(), file_name="images_jie_final_2/{}_image".format(str.zfill(str(i), 3)))
