# Txt2Fix

Light wrapper for styleCLIP latent optimization for image revision.

## 1. Requirements 
- Currently supports face-centered image, with revision restricted to face domain
- [Poetry](https://python-poetry.org/) : Dependency manager



### 1-1. Prepare Environment

- Download 2 pretrained stylegan2, face-enocder, shape_predictor, HyperStyle to project `txt2fix/models` directory
  - [Download pretrained StyleGan2](https://drive.google.com/file/d/1UC_22inUDEZiAfZ-UaQO_AZ4Ah40mAr8/view?usp=sharing) 
  - [Download pretrained FaceEncoder](https://drive.google.com/file/d/1BlHw_7pFxwCL51o6GKLyAwyIoqb9p0U2/view?usp=sharing)
  - [Download pretrained ShapePredictor](https://drive.google.com/file/d/1XRKtDDSQqug-OmYPbXWRjBCMfI2JmkQP/view?usp=sharing)
  - [Download pretrained HyperStyle](https://drive.google.com/file/d/1_5g-wkZQ3QmMD3uo0nJzlwTzX9mSkN67/view?usp=drive_link)
  - [Download pretrained encoding4editing](https://drive.google.com/file/d/1ceyCq126bUqbGoakpwWVyt5AstvrWHih/view?usp=sharing)
  - [Download pretrained StyleGan2_Pkl](https://drive.google.com/file/d/1wNdsEFGyNaC_6WpP81mpYrfoMbpgtPP7/view?usp=sharing)
  - [Download pretrained BERT-ViT](https://drive.google.com/file/d/1jxf1TThQqdjYwk8wPZueP3eNeuz9WLu7/view?usp=sharing)


## Build & Run
```bash
# recommend python 3.10 or higher
poetry install
poetry run client
```

## Refernece

- [StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery](https://github.com/orpatashnik/StyleCLIP)
- [StyleCLIP-Tutorial](https://github.com/ndb796/StyleCLIP-Tutorial)