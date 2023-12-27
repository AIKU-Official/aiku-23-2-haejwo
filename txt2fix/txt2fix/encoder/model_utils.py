import torch
from argparse import Namespace
from txt2fix.config import model_paths

import sys
sys.path.extend(['.', '..'])

from .hyperstyle import HyperStyle
from .encoders.e4e import e4e


def load_model(checkpoint_path, device='cuda', update_opts=None, is_restyle_encoder=False):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    opts['checkpoint_path'] = checkpoint_path
    opts['load_w_encoder'] = True

    if update_opts is not None:
        if type(update_opts) == dict:
            opts.update(update_opts)
        else:
            opts.update(vars(update_opts))

    opts = Namespace(**opts)
    
    net = HyperStyle(opts)

    net.eval()
    net.to(device)
    return net, opts

def run_alignment(image_path):
    import dlib
    from .align_faces_parallel import align_face
    predictor = dlib.shape_predictor(model_paths['shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print(f"Finished running alignment on image: {image_path}")
    return aligned_image