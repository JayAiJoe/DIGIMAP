import os
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
import copy
from util import *
from PIL import Image

from model import *
import moviepy.video.io.ImageSequenceClip
import scipy
import kornia.augmentation as K

from base64 import b64encode
import gradio as gr
from torchvision import transforms

# torch.hub.download_url_to_file('https://i.imgur.com/HiOTPNg.png', 'mona.png')
# torch.hub.download_url_to_file('https://i.imgur.com/Cw8HcTN.png', 'painting.png')

device = 'cpu'
latent_dim = 8
n_mlp = 5
num_down = 3

G_A2B = Generator(256, 4, latent_dim, n_mlp, channel_multiplier=1, lr_mlp=.01,n_res=1).to(device).eval()

ensure_checkpoint_exists('GNR_checkpoint_full.pt')
ckpt = torch.load('GNR_checkpoint_full.pt', map_location=device)

G_A2B.load_state_dict(ckpt['G_A2B_ema'])

# mean latent
truncation = 1
with torch.no_grad():
    mean_style = G_A2B.mapping(torch.randn([1000, latent_dim]).to(device)).mean(0, keepdim=True)


test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
])
plt.rcParams['figure.dpi'] = 200

# torch.manual_seed(84986)

num_styles = 1
style = torch.randn([num_styles, latent_dim]).to(device)


def inference(input_im):
    if input_im == None:
        return
    real_A = test_transform(input_im).unsqueeze(0).to(device)

    with torch.no_grad():
        A2B_content, _ = G_A2B.encode(real_A)
        #fake_A2B = G_A2B.decode(A2B_content.repeat(num_styles,1,1,1), style)
        fake_A2B = G_A2B.decode(A2B_content.repeat(num_styles,1,1,1), torch.randn([num_styles, latent_dim]).to(device))
        std=(0.5, 0.5, 0.5)
        mean=(0.5, 0.5, 0.5)
        z = fake_A2B * torch.tensor(std).view(3, 1, 1)
        z = z + torch.tensor(mean).view(3, 1, 1)
        tensor_to_pil = transforms.ToPILImage(mode='RGB')(z.squeeze())
    return tensor_to_pil

def clear(image):
    return

def setsample(image):
    return image


with gr.Blocks() as demo:
    gr.Markdown("<h1>GANs N' Roses</h1>")
    gr.Markdown("""Convert real-life face images into diverse anime versions of themselves. Use the default sample image or replace the input
                by first clicking X then dragging a new image into the Input box. Crop the image by cliking the pen tool. Click <b>Run</b> to transform the input
                into an anime version. Click <b>Clear</b> to clear the ouput box. Try running it multiple times for different anime styles!""")
    
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", value ="sample_images/1.JPG", label="Input")
            with gr.Row():
                clr = gr.Button("Clear") #needs implementation
                run = gr.Button("Run")
        with gr.Column():
            out = gr.outputs.Image(type="pil")
        clr.click(fn=clear, inputs=inp, outputs=inp) # clear input gr.Image
        clr.click(fn=clear, inputs=out, outputs=out) # clear output gr.Image

          
    gr.Markdown("<h3>Sample Inputs</h3>")
   
    with gr.Row():
            with gr.Column():
                sample1 = gr.Image(value="sample_images/1.JPG")
            with gr.Column():
                samplebtn1 = gr.Button(value="Try sample 1")
                samplebtn1.click(fn=setsample, inputs=sample1, outputs=inp)

            with gr.Column():
                sample2 = gr.Image(value="sample_images/2.JPG")
            with gr.Column():
                samplebtn2 = gr.Button(value="Try sample 2")
                samplebtn2.click(fn=setsample, inputs=sample2, outputs=inp)    

            with gr.Column():
                sample3 = gr.Image(value="sample_images/3.JPG")
            with gr.Column():
                samplebtn3 = gr.Button(value="Try sample 3")
                samplebtn3.click(fn=setsample, inputs=sample3, outputs=inp)    
        
    #add info here
    gr.Markdown("""
                GANs N' Roses (GNR) is an image-to-image framework for face images that uses a multimodal approach with novel definitions for content and style.
                <b>Content</b> is defined as what changes when a augmentations are applied to a face image. <b>Style</b> is defined as what does not change when augmentations
                are applied to a face image.

                GNR's implementation borrows heavily from StyleGAN2; however, adversarial loss is derived from the introduced content and style definitions, ensuring diversity of
                outputs when repeatedly transforming the same input face image.

                The current implementation was trained on the selfie2anime dataset and transforms real human faces into anime faces. Due to limitations of the dataset, GNR works best
                when working with <b>female face inputs</b> that are <b>cropped to include only the face</b> (no neck and body).

                GNR was implemented by Chong, M. & Forsyth, D. (2021) in the paper GANs N' Roses: Stable, Controllable, Diverse Image to Image Translation (works for videos too!)
                """)

    
    run.click(fn=inference, inputs=inp, outputs=out)
  
demo.launch(share = True)
