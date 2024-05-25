import argparse
import os
from transformers import BertTokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb

from dataset.CramedDataset import CramedDataset
from dataset.SarcasmDataset import SarcasmDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import VADataset
from models.basic_model import AVClassifier,ITClassifier,Threelassifier
from utils.utils import setup_seed, weight_init, remove_module_prefix,CELoss
from dataset.NvGesture import NvGestureDataset


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Twitter15', type=str,
                        help='nvGesture, kinetics_sound, CREMAD, AVE, Sarcasm')
    parser.add_argument('--type', default='Image_Text', type=str,
                        help='Video_Visual, Image_Text, Three')
    parser.add_argument('--modulation', default='Normal', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film','constractive'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=4, type=int)
    parser.add_argument('--audio_path', default='./dataset/CREMA/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./dataset/CREMA/', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=50, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=20, type=int, help='where modulation ends')
    parser.add_argument('--alpha', type=float,default=0.1, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', type=str,default='./ckpt/Twitter15/concat/', help='path to save trained models')
    parser.add_argument('--train', type=str,default=True, help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=156, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')

    return parser.parse_args()


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = CELoss(label_smooth=0.05, class_num=31)
    # criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    for step, (spec, image, label) in enumerate(dataloader):

        #pdb.set_trace()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        if args.fusion_method  == 'constractive':
            a, v, out, itmloss = model(spec.unsqueeze(1).float(), image.float())
        else:
            a, v, out = model(spec.unsqueeze(1).float(), image.float())

        if args.fusion_method == 'sum':
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                     model.module.fusion_module.fc_y.bias)
            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                     model.module.fusion_module.fc_x.bias)
        else:
            weight_size = model.module.fusion_module.fc_out.weight.size(1)
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                     + model.module.fusion_module.fc_out.bias / 2)

            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                     + model.module.fusion_module.fc_out.bias / 2)

        loss = criterion(out, label)
        if args.fusion_method  == 'constractive':
            loss += itmloss

        
        loss_v = criterion(out_v, label)
        # loss += loss_v 
        loss_a = criterion(out_a, label)
        # loss += loss_a 
        loss.backward()

        if args.modulation == 'Normal':
            # no modulation, regular optimization
            pass
        else:
            # Modulation starts here !
            score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])

            ratio_v = score_v / score_a
            ratio_a = 1 / ratio_v

            """
            Below is the Eq.(10) in our CVPR paper:
                    1 - tanh(alpha * rho_t_u), if rho_t_u > 1
            k_t_u =
                    1,                         else
            coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """

            if ratio_v > 1:
                coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
                coeff_a = 1
            else:
                coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
                coeff_v = 1

            if args.modulation_starts <= epoch <= args.modulation_ends: # bug fixed
                for name, parms in model.named_parameters():
                    layer = str(name).split('.')[1]

                    if 'audio' in layer and len(parms.grad.size()) == 4:
                        if args.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_a + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif args.modulation == 'OGM':
                            parms.grad *= coeff_a

                    if 'visual' in layer and len(parms.grad.size()) == 4:
                        if args.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_v + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif args.modulation == 'OGM':
                            parms.grad *= coeff_v
            else:
                pass


        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

    scheduler.step()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'kinetics_sound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            if args.fusion_method  == 'constractive':
                a, v, out,_  = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec.unsqueeze(1).float(), image.float())

            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                         model.module.fusion_module.fc_y.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                         model.module.fusion_module.fc_x.bias / 2)
            else:
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)

            prediction = softmax(out)

            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                #pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)



def train_epoch_Image_Text(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = CELoss(label_smooth=0, class_num=3)
    # criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()


    tokenizer = BertTokenizer.from_pretrained('./modelPth/bert')
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0
    loss_list = []
    for step, (image, text, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(image.device)
        input_list = []
        input_list.append(text_input),input_list.append(image)


        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        a, v, out, itmloss = model(input_list)
        # a, v, out = model(input_list)

        if args.fusion_method == 'sum':
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                     model.module.fusion_module.fc_y.bias)
            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                     model.module.fusion_module.fc_x.bias)
        else:
            weight_size = model.module.fusion_module.fc_out.weight.size(1)
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                     + model.module.fusion_module.fc_out.bias / 2)

            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                     + model.module.fusion_module.fc_out.bias / 2)
        epoch_tensor = torch.tensor(epoch)


        loss =criterion(out, label)

        # loss = (1 - torch.exp((-1) * 1.0 / (epoch_tensor + 1))) * criterion(out, label)
        # loss += torch.exp((-1) * 1.0 / (epoch_tensor + 1)) *  itmloss

        
        loss_v = criterion(out_v, label)
        # loss += loss_v 
        loss_a = criterion(out_a, label)
        # loss += loss_a 
        loss.backward()

        if args.modulation == 'Normal':
            # no modulation, regular optimization
            pass
        else:
            # Modulation starts here !
            score_v = sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
            score_a = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])

            ratio_v = score_v / score_a
            ratio_a = 1 / ratio_v

            """
            Below is the Eq.(10) in our CVPR paper:
                    1 - tanh(alpha * rho_t_u), if rho_t_u > 1
            k_t_u =
                    1,                         else
            coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """

            if ratio_v > 1:
                coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
                coeff_a = 1
            else:
                coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
                coeff_v = 1


            if args.modulation_starts <= epoch <= args.modulation_ends: # bug fixed
                for name, parms in model.named_parameters():
                    layer = str(name).split('.')[1]

                    if 'visual' in layer and len(parms.grad.size()) == 4:
                        if args.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_v + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif args.modulation == 'OGM':
                            parms.grad *= coeff_v

                    if 'text' in layer and len(parms.grad.size()) == 4:
                        if args.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_v + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif args.modulation == 'OGM':
                            parms.grad *= coeff_v
            else:
                pass


        optimizer.step()


        # print()
        # print(loss.item())
        loss_list.append(loss.item())
        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

    scheduler.step()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), loss_list


def valid_Image_Text(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)
    tokenizer = BertTokenizer.from_pretrained('./modelPth/bert')

    if args.dataset == 'Sarcasm':
        n_classes = 2
    elif args.dataset == 'Twitter15':
        n_classes = 3
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

    for step, (image, text, label) in enumerate(dataloader):
            image = image.to(device)
            label = label.to(device)
            text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(image.device)
            input_list = []
            input_list.append(text_input),input_list.append(image)



            # TODO: make it simpler and easier to extend
            a, v, out, _ = model(input_list)


            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                         model.module.fusion_module.fc_y.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                         model.module.fusion_module.fc_x.bias / 2)
            else:
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)

            prediction = softmax(out)

            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                #pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def train_epoch_Three(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = CELoss(label_smooth=0.5, class_num=25)
    # criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()

    print("Start training ... ")

    _loss = 0
    _loss_r = 0
    _loss_o = 0
    _loss_d = 0

    for step,(rgb,of,depth, label) in enumerate(dataloader):
        rgb = rgb.to(device)
        of = of.to(device)
        depth = depth.to(device)
        label = label.to(device)

        input_list = []
        input_list.append(rgb), input_list.append(of), input_list.append(depth)


        optimizer.zero_grad()


        rgb_feat, of_feat, depth_feat, out = model(input_list)


        weight_size = model.module.fusion_module.fc_out.weight.size(1)
        out_r = (torch.mm(rgb_feat, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 3], 0, 1))
                     + model.module.fusion_module.fc_out.bias / 3)

        out_o = (torch.mm(of_feat, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 3:weight_size*2 // 3], 0, 1))
                     + model.module.fusion_module.fc_out.bias / 3)

        out_d = (torch.mm(depth_feat, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size*2 // 3:], 0, 1))
                + model.module.fusion_module.fc_out.bias / 3)

        loss = criterion(out, label)
        # loss += itmloss

        # loss = (1 - torch.exp((-1) * 1.0 / (epoch_tensor + 1))) * criterion(out, label)
        # loss += torch.exp((-1) * 1.0 / (epoch_tensor + 1)) *  itmloss

        loss_r = criterion(out_r, label)
        # loss += loss_r
        loss_o = criterion(out_o, label)
        # loss += loss_o 
        loss_d = criterion(out_d, label)
        # loss += loss_d 


        loss.backward()

    
        optimizer.step()

        _loss += loss.item()

        _loss_r += loss_r.item()
        _loss_o += loss_o.item()
        _loss_d += loss_d.item()

    scheduler.step()

    return _loss / len(dataloader), _loss_r / len(dataloader), _loss_o / len(dataloader), _loss_d / len(dataloader)


def valid_Three(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'nvGesture':
        n_classes = 25
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_r = [0.0 for _ in range(n_classes)]
        acc_o = [0.0 for _ in range(n_classes)]
        acc_d = [0.0 for _ in range(n_classes)]


        for step,(rgb,of,depth, label) in enumerate(dataloader):
            rgb = rgb.to(device)
            of = of.to(device)
            depth = depth.to(device)

            input_list = []
            input_list.append(rgb), input_list.append(of), input_list.append(depth)


            rgb_feat, of_feat, depth_feat, out = model(input_list)


            weight_size = model.module.fusion_module.fc_out.weight.size(1)
            out_r = (torch.mm(rgb_feat, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 3], 0, 1))
                        + model.module.fusion_module.fc_out.bias / 3)

            out_o = (torch.mm(of_feat, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 3:weight_size*2 // 3], 0, 1))
                        + model.module.fusion_module.fc_out.bias / 3)

            out_d = (torch.mm(depth_feat, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size*2 // 3:], 0, 1))
                    + model.module.fusion_module.fc_out.bias / 3)

            prediction = softmax(out)

            pred_r = softmax(out_r)
            pred_o = softmax(out_o)
            pred_d = softmax(out_d)

            for i in range(rgb.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                r = np.argmax(pred_r[i].cpu().data.numpy())
                o = np.argmax(pred_o[i].cpu().data.numpy())
                d = np.argmax(pred_d[i].cpu().data.numpy())
                num[label[i]] += 1.0

                #pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == r:
                    acc_r[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == o:
                    acc_o[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == d:
                    acc_d[label[i]] += 1.0

    return sum(acc) / sum(num), sum(acc_r) / sum(num), sum(acc_o) / sum(num), sum(acc_d) / sum(num)

def main():
    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')
    if args.type == "Video_Visual":
        model = AVClassifier(args)
    elif args.type == "Image_Text":
        model = ITClassifier(args)
    else:
        model = Threelassifier(args)
        
    
    # loaded_dict = torch.load("./ckpt/Twitter15/acc_0.7348.pth")
    # state_dict = loaded_dict['model']
    # state_dict = remove_module_prefix(state_dict)

    # model.load_state_dict(state_dict)    

    # model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'kinetics_sound':
        train_dataset = VADataset(args, mode='train')
        test_dataset = VADataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == "Sarcasm":
        train_dataset = SarcasmDataset(ann_file= "./dataset/Sarcasm/annotations/train.tsv", type="train",image_root="./dataset/Sarcasm/images")
        test_dataset = SarcasmDataset(ann_file= "./dataset/Sarcasm/annotations/test.tsv", type="test",image_root="./dataset/Sarcasm/images")
    elif args.dataset == "Twitter15":
        train_dataset = SarcasmDataset(ann_file= "./dataset/Twitter15/annotations/train.tsv", type="train",image_root="./dataset/Twitter15/twitter2015_images")
        test_dataset = SarcasmDataset(ann_file= "./dataset/Twitter15/annotations/test.tsv", type="test",image_root="./dataset/Twitter15/twitter2015_images")
    elif args.dataset == "nvGesture":
        train_dataset = NvGestureDataset(mode='train')
        test_dataset = NvGestureDataset(mode='test')

    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)

    if args.train:

        best_acc = 0.0

        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))

            if args.type == "Video_Visual":
                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                        train_dataloader, optimizer, scheduler)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
            elif args.type == "Image_Text":
                batch_loss, batch_loss_a, batch_loss_v, loss_list= train_epoch_Image_Text(args, epoch, model, device,
                                                                        train_dataloader, optimizer, scheduler)
                print(loss_list)
                acc, acc_a, acc_v = valid_Image_Text(args, model, device, test_dataloader)
            
            else:
                batch_loss, batch_loss_r, batch_loss_o, batch_loss_d= train_epoch_Three(args, epoch, model, device,
                                                                        train_dataloader, optimizer, scheduler)
                acc, acc_r, acc_o, acc_d = valid_Three(args, model, device, test_dataloader)


            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'epoch_{}_acc_{}.pth'.format(epoch, acc)

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)

                if args.type == "Video_Visual":
                    print('The best model has been saved at {}.'.format(save_dir))
                    print("Loss: {:.4f}, Acc: {:.4f}".format(batch_loss, acc))
                    print("Audio Acc: {:.4f}， Visual Acc: {:.4f} ".format(acc_a, acc_v))
                elif args.type =="Image_Text":
                    print('The best model has been saved at {}.'.format(save_dir))
                    print("Loss: {:.4f}, Acc: {:.4f}".format(batch_loss, acc))
                    print(" Text Acc: {:.4f}， Image Acc: {:.4f} ".format(acc_a, acc_v))
                elif args.type =="Three":
                    print('The best model has been saved at {}.'.format(save_dir))
                    print("Loss: {:.4f}, Acc: {:.4f}".format(batch_loss, acc))
                    print("RGB Acc: {:.4f}， Of Acc: {:.4f} , Depth Acc:{}".format(acc_r, acc_o, acc_d))

            else:
                if args.type == "Video_Visual":
                    print("Loss: {:.4f}, Acc: {:.4f}".format(batch_loss, acc))
                    print("Audio Acc: {:.4f}， Visual Acc: {:.4f} ".format(acc_a, acc_v))
                elif args.type =="Image_Text":
                    print("Loss: {:.4f}, Acc: {:.4f}".format(batch_loss, acc))
                    print(" Text Acc: {:.4f}， Image Acc: {:.4f} ".format(acc_a, acc_v))
                elif args.type =="Three":
                    print("Loss: {:.4f}, Acc: {:.4f}".format(batch_loss, acc))
                    print("RGB Acc: {:.4f}， Of Acc: {:.4f} , Depth Acc:{:.4f}".format(acc_r, acc_o, acc_d))

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model = model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
