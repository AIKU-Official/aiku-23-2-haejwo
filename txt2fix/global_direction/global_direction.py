from argparse import Namespace
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import clip

from ..utils import tensor2im
from ..encoder.encoders.psp import pSp
from ..config import IMAGENET_TEMPLATES, device, model_paths, fs3_file_path
from .manipulate import Manipulator
from .util import GetDt, GetBoundary


EXPERIMENT_ARGS = {}
EXPERIMENT_ARGS['transform'] = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

resize_dims = (256, 256)

def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return images, latents

def save_image_and_get_path(input_image):
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image) # numpy array to pil

    input_image = input_image.convert("RGB")

    image_path = '/content/uploaded_image.png'
    input_image.save(image_path)

    return image_path

def multilingual_global(input_image, neutral, target, beta, alpha):
    ckpt = torch.load(model_paths['e4e_ffhq_model_path'], map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_paths['e4e_ffhq_model_path']
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print('e4e successfully loaded!')
    
    model, preprocess = clip.load("ViT-B/32", device=device,jit=False)

    dataset_name='ffhq'
    if not os.path.exists("latent.pt"):
        return  
    transformed_image = torch.load("latent.pt").reshape(1,18,512).to(device)
    
    
    network_pt=model_paths['stylegan_ffhq_pkl']
    M=Manipulator()
    M.device=device
    G=M.LoadModel(network_pt,device)
    M.G=G
    M.SetGParameters()
    num_img=100_000
    M.GenerateS(num_img=num_img)
    M.GetCodeMS()
    np.set_printoptions(suppress=True)

    fs3=np.load(fs3_file_path)


    # Run the image through the model
    # with torch.no_grad():
    #     images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
    #     result_image, latent = images[0], latents[0]
    # torch.save(latents, 'asd.pt')

    # original image
    img_index = 0
    latents=torch.load('latent.pt').reshape(1,18,512)
    dlatents_loaded=M.G.synthesis.W2S(latents)

    img_indexs=[img_index]
    dlatents_loaded=M.S2List(dlatents_loaded)

    dlatent_tmp=[tmp[img_indexs] for tmp in dlatents_loaded]

    M.num_images=len(img_indexs)

    M.alpha=[0]
    M.manipulate_layers=[0]
    codes,out=M.EditOneC(0,dlatent_tmp)
    original=Image.fromarray(out[0,0]).resize((512,512))
    M.manipulate_layers=None


    classnames = [target, neutral]
    dt = GetDt(classnames, model, IMAGENET_TEMPLATES)

    M.alpha = [alpha]
    boundary, c = GetBoundary(fs3, dt, M, threshold=beta)
    codes = M.MSCode(dlatent_tmp, boundary)
    out = M.GenerateImg(codes)
    output_image = Image.fromarray(out[0,0])

    return output_image