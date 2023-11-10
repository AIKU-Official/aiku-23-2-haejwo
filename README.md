# Txt2Fix

light wrapper for styleCLIP latent optimization for image revision.

## Requirements 
- Currently supports face-centered image, with revision restricted to face domain

## Prepare Environment

- Download 2 pretrained stylegan2, face-enocder to project directory
  - [Download pretrained StyleGan2](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view) 
  - [Download pretrained FaceEncoder](https://drive.google.com/file/d/1M-hsL3W_cJKs77xM1mwq2e9-J0_m7rHP/view?usp=sharing)
- Credits : [HyperStyle](https://github.com/yuval-alaluf/hyperstyle/)


## Build & Run
```bash
# recommend python 3.10 or higher
# MY_HOSTNAME=0.0.0.0
# MY_PORT=9000
pip install -r requirements.txt 
python client.py --host=MY_HOSTNAME --port=MY_PORT
```

## Refernece
- [HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing](https://github.com/yuval-alaluf/hyperstyle/)
- [StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery](https://github.com/orpatashnik/StyleCLIP)
- [StyleCLIP-Tutorial](https://github.com/ndb796/StyleCLIP-Tutorial)