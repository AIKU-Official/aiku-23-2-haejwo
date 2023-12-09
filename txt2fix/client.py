
# %%
import gradio as gr
import time
import requests
from PIL import Image
import io
from txt2fix.styleclip import inferLatentOptim
from txt2fix.stylegan_encode import styleganEncode
import torch

app = gr.Blocks()

with app:
    with gr.Row():
        gr.Markdown('''
                    # 보정..해줘!
                    ''')
    with gr.Row():
        gr.Markdown('''
                    # 1. StyleGAN은 당신을 이렇게 생각해요
                    StyleGAN latent로 변환
                    ''')
    
    with gr.Row():
        with gr.Column(scale=1, ):
            input_image_t = gr.Image(type="pil", label="Input Image")
            con_button_t = gr.Button("\"해줘\"")
        with gr.Column(scale=1):
            output_image_t = gr.Image(type="pil", label="Output Image")
            recent_latents= gr.List([])
    with gr.Row():
        gr.Markdown('''
                    # 2. 보정 시작
                    StyleGAN latent Optimization
                    ''')
    with gr.Row():
        with gr.Column(scale=1, ):
            input_image = gr.Image(type="pil", label="Input Image", value=None)
            input_prompt = gr.Textbox(lines=1, interactive=True,label="Prompt")
            steps = gr.Slider(minimum=10, maximum=200, label="Steps")
            clip_loss_strength = gr.Slider(minimum=0, maximum=5, value=float ,label="Clip Loss Strength")
            gen_button = gr.Button("\"해줘\"")
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Output Image")

    gen_button.click(fn=inferLatentOptim, inputs=[input_prompt, steps, clip_loss_strength, input_image], outputs=output_image)
    con_button_t.click(fn=styleganEncode, inputs=[input_image_t], outputs=[output_image_t,recent_latents])

# TODO : Encode Resueable
# - global var
# - main.py = encoder connect function
# - Dockerize
# - Dependency cleanup


