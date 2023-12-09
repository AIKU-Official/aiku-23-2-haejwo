from argparse import Namespace
import os
import torch

args = Namespace()
args.lr_rampup = 0.05
args.lr = 0.1
args.step = 150
args.l2_lambda = 0.005 # The weight for similarity to the original image.
args.save_intermediate_image_every = 1
args.results_dir = 'results'
args.stylegan2 = 'txt2fix/models/stylegan2-ffhq-config-f.pt'
args.w_face_enocder = 'faces_w_encoder.pt'


# CUDA -> MPS/device
# GPU accleration configuration
if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")

os.makedirs(args.results_dir, exist_ok=True)

model_paths = {
	# models for backbones and losses
	'ir_se50': 'txt2fix/models/model_ir_se50.pth',
	'resnet34': 'txt2fix/models/resnet34-333f7ec4.pth',
	'moco': 'txt2fix/models/moco_v2_800ep_pretrain.pt',
	# stylegan2 generators
	'stylegan_ffhq': 'txt2fix/models/stylegan2-ffhq-config-f.pt',
	'stylegan_cars': 'txt2fix/models/stylegan2-car-config-f.pt',
	'stylegan_ada_wild': 'txt2fix/models/afhqwild.pt',
	# model for face alignment
	'shape_predictor': 'txt2fix/models/shape_predictor_68_face_landmarks.dat',
	# models for ID similarity computation
	'curricular_face': 'txt2fix/models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'txt2fix/models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'txt2fix/models/mtcnn/rnet.npy',
	'mtcnn_onet': 'txt2fix/models/mtcnn/onet.npy',
	# WEncoders for training on various domains
	'faces_w_encoder': 'txt2fix/models/faces_w_encoder.pt',
	'cars_w_encoder': 'txt2fix/models/cars_w_encoder.pt',
	'afhq_wild_w_encoder': 'txt2fix/models/afhq_wild_w_encoder.pt',
	# models for domain adaptation
	'restyle_e4e_ffhq': 'txt2fix/models/restyle_e4e_ffhq_encode.pt',
	'stylegan_pixar': 'txt2fix/models/pixar.pt',
	'stylegan_toonify': 'txt2fix/models/ffhq_cartoon_blended.pt',
	'stylegan_sketch': 'txt2fix/models/sketch.pt',
	'stylegan_disney': 'txt2fix/models/disney_princess.pt',
	'shape_predictor':  'txt2fix/models/shape_predictor_68_face_landmarks.dat',
 	"hyperstyle_model_path": "txt2fix/models/hyperstyle_ffhq.pt"
}

RESNET_MAPPING = {
    'layer1.0': 'body.0',
    'layer1.1': 'body.1',
    'layer1.2': 'body.2',
    'layer2.0': 'body.3',
    'layer2.1': 'body.4',
    'layer2.2': 'body.5',
    'layer2.3': 'body.6',
    'layer3.0': 'body.7',
    'layer3.1': 'body.8',
    'layer3.2': 'body.9',
    'layer3.3': 'body.10',
    'layer3.4': 'body.11',
    'layer3.5': 'body.12',
    'layer4.0': 'body.13',
    'layer4.1': 'body.14',
    'layer4.2': 'body.15',
}