# %%
import torch
from stylegan2.model import Generator
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage
import clip
import os
from utils import get_lr
from clip_loss import CLIPLoss
from config import args, device
from torchsummary import summary

g_ema = Generator(1024, 512, 8)
g_ema.load_state_dict(torch.load(args.stylegan2)["g_ema"], strict=False)
g_ema.eval()
g_ema = g_ema.to(device)

# %%
# sample image from stylegan2
mean_latent = g_ema.mean_latent(4096)

latent_code_init_not_trunc = torch.randn(1, 512).to(device)
with torch.no_grad():
    img_orig, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
                                truncation=0.7, truncation_latent=mean_latent)

# Visualize a random latent vector.
image = ToPILImage()(make_grid(img_orig.detach().cpu(), normalize=True, scale_each=True, value_range=(-1, 1), padding=0))
h, w = image.size
image.resize((h // 2, w // 2))
#%%
text_inputs = torch.cat([clip.tokenize(args.description)]).to(device)

# Initialize the latent vector to be updated.
latent = latent_code_init.detach().clone()
latent.requires_grad = True

clip_loss = CLIPLoss()
optimizer = torch.optim.Adam([latent], lr=args.lr)

# %%
for i in range(args.step):
    # Adjust the learning rate.
    t = i / args.step
    lr = get_lr(t, args.lr)
    optimizer.param_groups[0]["lr"] = lr

    # Generate an image using the latent vector.
    img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
    # Calculate the loss value.
    c_loss = clip_loss(img_gen, text_inputs)
    l2_loss = ((latent_code_init - latent) ** 2).sum()
    loss = c_loss + args.l2_lambda * l2_loss

    # Get gradient and update the latent vector.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log the current state.
    if (i+1)%10 == 0:
      print(f"step: {i+1}, lr: {lr}, loss: {loss.item():.4f}")
    if args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
        with torch.no_grad():
            img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
        save_image(img_gen, f"results/{str(i).zfill(5)}.png", normalize=True, value_range=(-1, 1))

with torch.no_grad():
    img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

# Display the initial image and result image.
final_result = torch.cat([img_orig, img_gen])
save_image(final_result.detach().cpu(), os.path.join(args.results_dir, "final_result.jpg"), normalize=True, scale_each=True, value_range=(-1, 1))
