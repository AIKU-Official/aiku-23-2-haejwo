# 보정..해줘

📢 2023년-2 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

## 소개

StyleCLIP을 이용한 얼굴 이미지 보정 데모 프로젝트 입니다.

## 방법론

- **문제** : 이미지 편집을 텍스트를 통해 보다 쉽게, 불필요한 부분이 변형되는 것을 최소화하자
- **방법론** : StyleCLIP, StyleGAN 과 CLIP을 이용해, 원본 이미지를 유지한 채, 텍스트로 쉽게 이미지를 편집하는 모델
  1. StyleGAN Inversion : 사용자 입력 이미지를 StyleGAN space의 latent 이미지 벡터로 변환
  2. StyleCLIP 
    - Latent-Optimization : CLIP loss, GAN Inception Loss 이용해 이미지 벡터를 최적화
    - Global Direction : CLIP space와 StyleGAN space direction을 맵핑하여, 이미지 벡터 최적화

## 환경 설정

### Environment Setup

- [Poetry](https://python-poetry.org/) : Dependency manager
- Download 2 pretrained stylegan2, face-enocder, shape_predictor, HyperStyle to project `txt2fix/models` directory
  - [Download pretrained StyleGan2](https://drive.google.com/file/d/1UC_22inUDEZiAfZ-UaQO_AZ4Ah40mAr8/view?usp=sharing) 
  - [Download pretrained FaceEncoder](https://drive.google.com/file/d/1BlHw_7pFxwCL51o6GKLyAwyIoqb9p0U2/view?usp=sharing)
  - [Download pretrained ShapePredictor](https://drive.google.com/file/d/1XRKtDDSQqug-OmYPbXWRjBCMfI2JmkQP/view?usp=sharing)
  - [Download pretrained HyperStyle](https://drive.google.com/file/d/1_5g-wkZQ3QmMD3uo0nJzlwTzX9mSkN67/view?usp=drive_link)
  - [Download pretrained encoding4editing](https://drive.google.com/file/d/1ceyCq126bUqbGoakpwWVyt5AstvrWHih/view?usp=sharing)
  - [Download pretrained StyleGan2_Pkl](https://drive.google.com/file/d/1wNdsEFGyNaC_6WpP81mpYrfoMbpgtPP7/view?usp=sharing)
  - [Download pretrained BERT-ViT](https://drive.google.com/file/d/1jxf1TThQqdjYwk8wPZueP3eNeuz9WLu7/view?usp=sharing)




## 사용 방법

### Build & Run
```bash
# recommend python 3.10 or higher
poetry install
poetry run client
```
## 예시 결과
- **Latent Optimization** : "Really sad face"
    ![example_image](https://github.com/AIKU-Official/aiku-23-2-haejwo/assets/55953815/f4f7b1ce-2feb-4005-b74f-9d97aa9b9b2c)

## 팀원

- [김민성](https://github.com/mingsung-k): StyleCLIP - LatentMapper/Global-Direction
- [김규민](https://github.com/KY00KIM/): StyleCLIP - Latent-Optimization
- [황정현]([홍길동의 github link](https://github.com/imjunghyunee/)): StyleCLIP - LatentMapper/Global-Direction

