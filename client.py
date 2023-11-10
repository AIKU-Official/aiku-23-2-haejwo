import gradio as gr
import time
import requests
from PIL import Image
import io
import argparse


app = gr.Blocks()

def generate_image(image, prompt,progress=gr.Progress()):
    progress(0, desc="Starting...")
    num_steps = 10
    
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
            input_image = gr.Image(type="pil", label="InputImage")
            input_prompt = gr.Textbox(lines=1, interactive=True,label="Prompt")
            gen_button = gr.Button("\"해줘\"")
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="OutputImage")

    gen_button.click(fn=generate_image, inputs=[input_image, input_prompt], outputs=output_image)


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio app")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Port to run app on.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run app on.")
    args = parser.parse_args()

    app.queue().launch(server_name=args.host, server_port=args.port)
    

if __name__ == "__main__":
    
    main()
