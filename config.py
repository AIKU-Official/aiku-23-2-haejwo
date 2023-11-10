from argparse import Namespace
import os
import torch

args = Namespace()
args.description = 'A really sad face'
args.lr_rampup = 0.05
args.lr = 0.1
args.step = 150
args.l2_lambda = 0.005 # The weight for similarity to the original image.
args.save_intermediate_image_every = 1
args.results_dir = 'results'
args.stylegan2 = 'stylegan2-ffhq-config-f.pt'
args.w_face_enocder = 'faces_w_encoder.pt'


# CUDA -> MPS/device
# GPU accleration configuration
if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")

os.makedirs(args.results_dir, exist_ok=True)
