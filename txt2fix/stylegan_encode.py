#%%
import torchvision.transforms as transforms
from PIL import Image
from txt2fix.encoder.model_utils import load_model, run_alignment
from .config import model_paths
from .encoder.inference_utils import run_inversion
from .utils import tensor2im
import torch
import time

def styleganEncode(image: Image) -> (Image, torch.Tensor):
    IMAGE_PATH = "./tmp.png"
    tic = time.time()
    hyperstyle_model_path = model_paths['hyperstyle_model_path']
    net, opts = load_model(hyperstyle_model_path, update_opts={"w_encoder_checkpoint_path": model_paths['faces_w_encoder']})
    toc = time.time()
    print('Hyperstyle load took {:.4f} seconds.'.format(toc - tic))
    opts.resize_outputs = False 

    image.convert("RGB").save(IMAGE_PATH)
    input_image = run_alignment(IMAGE_PATH)

    img_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    transformed_image = img_transforms(input_image) 

    with torch.no_grad():
        tic = time.time()
        result_batch, result_latents, tmp,_ = run_inversion(transformed_image.unsqueeze(0).cuda(), 
                                                        net, 
                                                        opts)
        toc = time.time()
        print('Encode took {:.4f} seconds.'.format(toc - tic))
    del net
    torch.cuda.empty_cache()
    torch.save(result_latents[0], "latent.pt")
    return tensor2im(result_batch[0]), [result_latents[0]]

