import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image

from utils import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp




import torch
import cv2
import os
import argparse
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from transformers import BertTokenizer
import sys
sys.path.append("..")
from models.basic_model import ITClassifier
def remove_module_prefix(state_dict):
    new_state_dict = type(state_dict)()
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def get_arguments(path, method):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Twitter15', type=str,
                        help='VGGSound, kinetics_sound, CREMAD, AVE')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--modulation', default='Normal', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default=f'{method}', type=str,
                        choices=['sum', 'concat', 'gated', 'film','constractive'])
    parser.add_argument('--ckpt_path', type=str,default=f'{path}', help='path to save trained models')
    parser.add_argument('--gpu_ids', default='2', type=str, help='GPU ids')
    parser.add_argument('--audio_path', default='./paper/dataset/CREMA/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./paper/dataset/CREMA/', type=str)
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=5, type=int)
    return parser.parse_args()


def getmodel(path, method):

    args = get_arguments(path, method)
    model = ITClassifier(args)
    loaded_dict = torch.load(args.ckpt_path)
    state_dict = loaded_dict['model']
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    return model.visual_encoder

resnet_1 = getmodel("", "concat")
resnet_1.eval()

resnet_2 = getmodel("", "concat")
resnet_2.eval()

resnet_3 = getmodel("", "concat")
resnet_3.eval()


resnet_4 = getmodel("", "constractive")
resnet_4.eval()

resnet_5 = getmodel("", "constractive")
resnet_5.eval()

resnet_6 = getmodel("", "constractive")
resnet_6.eval()

cam_dict = dict()



resnet_1_model_dict = dict(type='resnet', arch=resnet_1, layer_name='layer4', input_size=(224, 224))
resnet_1_gradcam = GradCAM(resnet_1_model_dict, True)
cam_dict['resnet_1'] =resnet_1_gradcam

resnet_2_model_dict = dict(type='resnet', arch=resnet_2, layer_name='layer4', input_size=(224, 224))
resnet_2_gradcam = GradCAM(resnet_2_model_dict, True)
cam_dict['resnet_2'] = resnet_2_gradcam

resnet_3_model_dict = dict(type='resnet', arch=resnet_3, layer_name='layer4', input_size=(224, 224))
resnet_3_gradcam = GradCAM(resnet_3_model_dict, True)
cam_dict['resnet_3'] = resnet_3_gradcam

resnet_4_model_dict = dict(type='resnet', arch=resnet_4, layer_name='layer4', input_size=(224, 224))
resnet_4_gradcam = GradCAM(resnet_4_model_dict, True)
cam_dict['resnet_4'] =resnet_4_gradcam

resnet_5_model_dict = dict(type='resnet', arch=resnet_5, layer_name='layer4', input_size=(224, 224))
resnet_5_gradcam = GradCAM(resnet_5_model_dict, True)
cam_dict['resnet_5'] = resnet_5_gradcam

resnet_6_model_dict = dict(type='resnet', arch=resnet_6, layer_name='layer4', input_size=(224, 224))
resnet_6_gradcam = GradCAM(resnet_6_model_dict, True)
cam_dict['resnet_6'] = resnet_6_gradcam


img_dir = "./dataset/Twitter15/twitter2015_images/1739565.jpg"
pil_img = PIL.Image.open(img_dir).convert('RGB')

normalizer = Normalize(mean=[0.48456249428840364, 0.4988762843596956, 0.530657871261656], std=[0.24398715912049204, 0.24868027118814742, 0.2573621194247526])
torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255)
torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
normed_torch_img = normalizer(torch_img)
sam = torch.stack([torch_img.squeeze().cpu()],0)
save_image(sam, "./same.pdf")


def imgae_plot(path, label, name):
    img_dir = path
    pil_img = PIL.Image.open(img_dir).convert('RGB')

    normalizer = Normalize(mean=[0.48456249428840364, 0.4988762843596956, 0.530657871261656], std=[0.24398715912049204, 0.24868027118814742, 0.2573621194247526])
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255)
    torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)


    images= []
    images_name = ['concat1','concat2','concat3','cons1','cons2','conc3']
    k = 0
    for gradcam in cam_dict.values():
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        print(result.shape)
        if k%3 ==0:
            images.append(torch.stack([torch_img.squeeze().cpu()],0))
        
        images.append(torch.stack([ result], 0))
        # print()
        # k=k+1
        image_single = torch.stack([ result], 0)
        save_image(image_single, f"./grad_{images_name[k]}.pdf")
        k=k+1

    images = make_grid(torch.cat(images, 0), nrow=4)
    save_image(images, f"./train/grad_{label}_{name}")

the packing has started . i just love moving ! 
import pandas as pd

data = pd.read_csv("./dataset/Twitter15/annotations/train.tsv", sep='\t')

for i in range(len(data)):
    path = "./dataset/Twitter15/twitter2015_images/"+data['ImageID'][i]
    label = data['Label'][i]
    imgae_plot(path, label, data['ImageID'][i])


