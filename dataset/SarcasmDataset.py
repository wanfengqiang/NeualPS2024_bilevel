import json
import os
import random
import re
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from PIL import ImageFile
from dataset.Randaugment import RandomAugment
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption


class SarcasmDataset(Dataset):
    def __init__(self, ann_file,type, image_root, max_words=30):

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            
        train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
        test_transform = transforms.Compose([
            transforms.Resize((224,224),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])   
        self.info = pd.read_csv(ann_file, sep='\t')
        if type=="train":
            self.transform = train_transform
        else:
            self.transform = test_transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}
        self.cls_num=2

    def __len__(self):
        return len(self.info)
    def get_num_classes(self):
        return self.cls_num
    def __getitem__(self, index):

        ann = self.info

        image_path = os.path.join(self.image_root, ann['ImageID'][index].split('/')[-1])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['String'][index], self.max_words)
        target=int(ann['Label'][index])
        return image,caption,target