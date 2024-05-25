import argparse
import os
from transformers import BertTokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.SarcasmDataset import SarcasmDataset
from torch.utils.tensorboard import SummaryWriter
import pdb

from sklearn.metrics import f1_score,average_precision_score

from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import VADataset
from models.basic_model import AVClassifier,ITClassifier
from utils.utils import setup_seed, weight_init


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, kinetics_sound, CREMAD, AVE')
    parser.add_argument('--type', default='Video_Visual', type=str,
                        help='Video_Visual, Image_Text, Three')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--modulation', default='Normal', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='constractive', type=str,
                        choices=['sum', 'concat', 'gated', 'film','constractive'])
    parser.add_argument('--ckpt_path', type=str,default='./ckpt/CRAMD/acc_0.8281.pth', help='path to save trained models')
    parser.add_argument('--gpu_ids', default='5', type=str, help='GPU ids')
    parser.add_argument('--audio_path', default='./dataset/CREMA/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./dataset/CREMA/', type=str)
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=5, type=int)
    return parser.parse_args()





def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'kinetics_sound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 2
    elif args.dataset == 'Twitter15':
        n_classes = 3
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))



    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        predictions = []
        predictions_a = []
        predictions_v= []
        labels = []  

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            # TODO: make it simpler and easier to extend
            if args.fusion_method == 'constractive':
                a, v, out,_  = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out  = model(spec.unsqueeze(1).float(), image.float())

            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                         model.fusion_module.fc_y.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                         model.fusion_module.fc_x.bias / 2)
            else:
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                         model.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                         model.fusion_module.fc_out.bias / 2)

            prediction = softmax(out)

            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):
                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                if np.asarray(label[i].cpu().data.numpy()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu().data.numpy()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu().data.numpy()) == a:
                    acc_a[label[i]] += 1.0

                predictions.append(prediction[i].cpu().data.numpy())
                predictions_a.append(pred_a[i].cpu().data.numpy())
                predictions_v.append(pred_v[i].cpu().data.numpy())
                labels.append(label[i].cpu().data.numpy())

        aps = []
        aps_a = []
        aps_v = []
        for i in range(n_classes):
            ap = average_precision_score((np.array(labels) == i).astype(int), np.array(predictions)[:, i])
            ap_a = average_precision_score((np.array(labels) == i).astype(int), np.array(predictions_a)[:, i])
            ap_v = average_precision_score((np.array(labels) == i).astype(int), np.array(predictions_v)[:, i])
            aps_a.append(ap_a)
            aps_v.append(ap_v)
            aps.append(ap)
    map_score = np.mean(aps)
    map_score_a = np.mean(aps_a)
    map_score_v = np.mean(aps_v)
    print("mAP for Audio (a): ", map_score_a)
    print("mAP for Video (v): ", map_score_v)
    print("mAP for Multi (M): ",map_score)


    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def valid_text(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)
    tokenizer = BertTokenizer.from_pretrained('./modelPth/bert')

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'kinetics_sound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'Sarcasm':
        n_classes = 2
    elif args.dataset == 'Twitter15':
        n_classes = 3
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        predictions = []  
        predictions_v = []  
        predictions_t = []  
        labels = []  

        for step, (image, text, label) in enumerate(dataloader):
            image = image.to(device)
            label = label.to(device)
            text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(image.device)
            input_list = []
            input_list.append(text_input),input_list.append(image)



            # TODO: make it simpler and easier to extend
            a, v, out, _ = model(input_list)

            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) +
                         model.fusion_module.fc_y.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                         model.fusion_module.fc_x.bias / 2)
            else:
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                         model.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                         model.fusion_module.fc_out.bias / 2)

            prediction = softmax(out)


            predicted_labels = prediction.argmax(dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
            labels.extend(label.cpu().numpy())

            pred_v = softmax(out_v)

            pred_v_labels = pred_v.argmax(dim=1)
            predictions_v.extend(pred_v_labels.cpu().numpy())


            pred_a = softmax(out_a)

            pred_t_labels = pred_a.argmax(dim=1)
            predictions_t.extend(pred_t_labels.cpu().numpy())

            for i in range(image.shape[0]):
                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                if np.asarray(label[i].cpu().data.numpy()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu().data.numpy()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu().data.numpy()) == a:
                    acc_a[label[i]] += 1.0


    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    print(f'All : Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions_v, average='macro')
    print(f'Image: Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')

    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions_t, average='macro')
    print(f'Text: Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')
    #     aps = []
    #     for i in range(n_classes):
    #         ap = average_precision_score((np.array(labels) == i).astype(int), np.array(predictions)[:, i])
    #         aps.append(ap)
    # map_score = np.mean(aps)
    # print(map_score)


    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)

def remove_module_prefix(state_dict):
    new_state_dict = type(state_dict)()
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict




def main():
    args = get_arguments()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')
    if args.type == "Video_Visual":
        model = AVClassifier(args)
    else:
        model = ITClassifier(args)

    

    if args.dataset == 'kinetics_sound':
        test_dataset = VADataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'Twitter15':
        test_dataset = SarcasmDataset(ann_file= "./dataset/Twitter15/annotations/test.tsv", type="test",image_root="./dataset/Twitter15/twitter2015_images")
    
    elif args.dataset == 'Sarcasm':
        test_dataset = SarcasmDataset(ann_file= "./dataset/Sarcasm/annotations/test.tsv", type="test",image_root="./dataset/Sarcasm/images")
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))


    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)

    
    loaded_dict = torch.load(args.ckpt_path)
    state_dict = loaded_dict['model']
    state_dict = remove_module_prefix(state_dict)



    model.load_state_dict(state_dict)
    model.to(device)
    print('Trained model loaded!')


    if args.type == "Video_Visual":
        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
    else:

        acc, acc_a, acc_v = valid_text(args, model, device, test_dataloader)
    print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
