import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion,Constractive,ConcatThree
from .Rodmodel import create_model
from transformers import BertModel
import torchvision
import torch.nn as nn
import torch.nn.init as init

class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        self.fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'kinetics_sound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if self.fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif self.fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif self.fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif self.fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        elif self.fusion == 'constractive':
            self.fusion_module = Constractive(output_dim=n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(self.fusion))

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

    
    def forward(self, audio, visual):
        # Your forward pass code remains the same
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        if self.fusion == 'constractive':
            a, v, out, itm = self.fusion_module(a, v)
            return a, v, out, itm
        else:
            a, v, out = self.fusion_module(a, v)
            return a, v, out


class ITClassifier(nn.Module):
    def __init__(self, args):
        super(ITClassifier, self).__init__()

        self.fusion = args.fusion_method
        if args.dataset == 'Sarcasm':
            n_classes = 2
        elif args.dataset == 'Twitter15':
            n_classes = 3
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if self.fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif self.fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif self.fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif self.fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        elif self.fusion == 'constractive':
            self.fusion_module = Constractive(output_dim=n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(self.fusion))

        self.text_encoder = BertModel.from_pretrained('../modelPth/bert', add_pooling_layer=False)
        self.visual_encoder = torchvision.models.resnet50()
        
        checkpoint = torch.load('../modelPth/resnet/resnet50.pth')
        self.visual_encoder.load_state_dict(checkpoint)

    
    def forward(self, input_list):
        # Your forward pass code remains the same
        text = input_list[0]
        image = input_list[1]

        text_embeds = self.text_encoder(text.input_ids,
                                                 attention_mask=text.attention_mask,
                                                 return_dict=True
                                                 ).last_hidden_state[:,0,:]

        image_embeds = self.visual_encoder(image)

        t = text_embeds
        i = image_embeds

        if self.fusion == 'constractive':
            t, i, out, itm = self.fusion_module(t, i)
            return t, i, out, itm
        else:
            t, i, out = self.fusion_module(t, i)
            return t, i, out,0 


class Threelassifier(nn.Module):
    def __init__(self, args):
        super(Threelassifier, self).__init__()

        self.fusion = args.fusion_method
        if args.dataset == 'nvGesture':
            n_classes = 25
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if self.fusion == 'concat':
            self.fusion_module = ConcatThree(output_dim=n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(self.fusion))
        

        self.encoder = create_model()


    
    def forward(self, input_list):
        # Your forward pass code remains the same
        rgb_feat, of_feat, depth_feat = self.encoder(input_list)

        if self.fusion == 'concat':
            rgb_feat, of_feat, depth_feat, out = self.fusion_module(rgb_feat, of_feat, depth_feat)
            return rgb_feat, of_feat, depth_feat, out
