
# %%
import gradio as gr
import time
import requests
from PIL import Image
import io
import argparse
from tt import bar
from model import inferLatentOptim

app = gr.Blocks()

def generate_image(image, prompt,progress=gr.Progress()):
    progress(0, desc="Starting...")
    num_steps = 10
    bar()
    print("input:", image, prompt)
    for i in progress.tqdm(range(num_steps), desc="Generating..."):
        time.sleep(0.1)
        
        progress = (i + 1) / num_steps

    response = requests.get("https://random.imagecdn.app/512/512")
    pil_image = Image.open(io.BytesIO(response.content))
    
    return pil_image

with app:
    with gr.Row():
        gr.Markdown('''
                    # 보정..해줘!
                    ......
                    ''')
    with gr.Row():
        with gr.Column(scale=1, ):
            input_image = gr.Image(type="pil", label="Input Image")
            input_prompt = gr.Textbox(lines=1, interactive=True,label="Prompt")
            steps = gr.Slider(minimum=10, maximum=200, label="Steps")
            clip_loss_strength = gr.Slider(minimum=0, maximum=5, value=float ,label="Clip Loss Strength")
            gen_button = gr.Button("\"해줘\"")
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Output Image")
    with gr.Row():
        gr.Markdown('''
                    # 변환..해줘!
                    ......
                    ''')
    with gr.Row():
        with gr.Column(scale=1, ):
            input_image_t = gr.Image(type="pil", label="Input Image")
            con_button_t = gr.Button("\"해줘\"")
        with gr.Column(scale=1):
            output_image_t = gr.Image(type="pil", label="Output Image")

    gen_button.click(fn=inferLatentOptim, inputs=[input_prompt, steps, clip_loss_strength], outputs=output_image)
    con_button_t.click(fn=generate_image, inputs=[], outputs=output_image_t)
    # gen_button.click(fn=generate_image, inputs=[input_image, input_prompt], outputs=output_image)



