import argparse
from .client import app

def main():
    parser = argparse.ArgumentParser(description="Launch Gradio app")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Port to run app on.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run app on.")
    args = parser.parse_args()

    app.queue().launch(server_name=args.host, server_port=args.port, share=True)

if __name__ == "__main__":
    main()