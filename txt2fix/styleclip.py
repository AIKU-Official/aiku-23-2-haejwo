# %%
import torch
from .stylegan2.model import Generator
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
import clip
import os
from .utils import get_lr
from .clip_loss import CLIPLoss
from .config import args, device
from tqdm import tqdm
from datetime import datetime 
from PIL import Image
import gradio as gr
from .stylegan_encode import styleganEncode
import time

def inferLatentOptim(text_prompt: str, steps: int, my_lmbda: float, input_image=None,progress=gr.Progress()):

    if os.path.exists("latent.pt"):
        latent_code_init = torch.load("latent.pt").reshape(1,18,512).to(device)
    tic = time.time()
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(args.stylegan2)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    toc = time.time()
    print('Stylegan load took {:.4f} seconds.'.format(toc - tic))

        # sample image from stylegan2
    if not os.path.exists("latent.pt"):
        mean_latent = g_ema.mean_latent(4096)    
        latent_code_init_not_trunc = torch.randn(1, 512).to(device)
        with torch.no_grad():
            img_orig, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
                                        truncation=0.7, truncation_latent=mean_latent)

    text_inputs = torch.cat([clip.tokenize(text_prompt)]).to(device)
    # Initialize the latent vector to be updated.
    latent = latent_code_init.detach().clone()
    latent.requires_grad = True

    clip_loss = CLIPLoss()
    optimizer = torch.optim.Adam([latent], lr=args.lr)
    tic = time.time()
    for i in progress.tqdm(range(steps), desc="Generating..."):
        # Adjust the learning rate.
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        # Generate an image using the latent vector.
        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
        # Calculate the loss value.
        c_loss = clip_loss(img_gen, text_inputs)
        l2_loss = ((latent_code_init - latent) ** 2).sum()
        loss = my_lmbda * c_loss + args.l2_lambda * l2_loss

        # Get gradient and update the latent vector.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    toc = time.time()
    print('Optimization took {:.4f} seconds.'.format(toc - tic))

    with torch.no_grad():
        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
        img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if input_image is None:
        final_result = torch.cat([img_orig, img_gen])
    else:
        final_result = img_gen
    save_image(final_result.detach().cpu(), os.path.join(args.results_dir, f"{now}.png"), normalize=True, value_range=(-1, 1))
    del g_ema
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return Image.open(os.path.join(args.results_dir, f"{now}.png"))
