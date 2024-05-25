import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.SarcasmDataset import SarcasmDataset
import pdb
import torch.nn.functional as F
from sklearn.metrics import f1_score,average_precision_score
import matplotlib.pyplot as plt
from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import VADataset
from models.basic_model import AVClassifier,ITClassifier
from utils.utils import setup_seed, weight_init


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, kinetics_sound, CREMAD, AVE')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--modulation', default='Normal', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='constractive', type=str,
                        choices=['sum', 'concat', 'gated', 'film','constractive'])
    parser.add_argument('--ckpt_path', type=str,default='./ckpt/CRAMD/acc_0.714.pth', help='path to save trained models')
    parser.add_argument('--gpu_ids', default='2', type=str, help='GPU ids')
    parser.add_argument('--audio_path', default='./dataset/CREMA/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='./dataset/CREMA/', type=str)
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=5, type=int)
    return parser.parse_args()





def plot(args, model, device, dataloader):


    with torch.no_grad():
        model.eval()
        a_feature= []
        b_feature= []
        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.unsqueeze(1).to(device)
            image = image.to(device)
            label = label.to(device)

            a = model.audio_net(spec)
            v = model.visual_net(image)
            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            a = F.adaptive_avg_pool2d(a, 1)
            v = F.adaptive_avg_pool3d(v, 1)

            a = torch.flatten(a, 1)
            v = torch.flatten(v, 1)


            a_norm = F.normalize(a, p=2, dim =-1)
            v_norm = F.normalize(v, p=2, dim =-1)
            a_feature.append(a_norm)
            b_feature.append(v_norm)
        


        all_img_features = torch.cat(a_feature, dim=0)
        all_text_features = torch.cat(b_feature, dim=0)


        def svd(X, n_components=2):
            # using SVD to compute eigenvectors and eigenvalues
            # M = np.mean(X, axis=0)
            # X = X - M
            U, S, Vt = np.linalg.svd(X)
            # print(S)
            return U[:, :n_components] * S[:n_components]
        
        all_img_features = all_img_features[50:100,:]
        print(all_img_features.shape)
        
        all_text_features = all_text_features[50:100,:]
        print(all_text_features.shape)

        torch.save(all_img_features, "./plot_data/concat/img_feature.pt")
        torch.save(all_text_features, "./plot_data/concat/text_feature.pt")

        distances = []
        for i in range(len(all_img_features)):
            img_feature = all_img_features[i]
            text_feature = all_text_features[i]
            distance = torch.norm(img_feature - text_feature).item()  
            distances.append(distance)

        # 计算平均距离
        average_distance = torch.tensor(distances).mean().item() 

        features_2d = svd(torch.cat([all_img_features, all_text_features], 0).cpu().detach().numpy())
        plt.figure(figsize=(5, 5))


        # 绘制图像特征散点图
        plt.scatter(features_2d[:-len(all_img_features), 0], features_2d[:-len(all_img_features), 1], c='orange', s=60, label='Audio', marker='o')

        # 绘制文本特征散点图
        plt.scatter(features_2d[-len(all_img_features):, 0], features_2d[-len(all_img_features):, 1], c='blue', s=60, label='Image', marker='x')

        # 绘制连接线
        for i in range(len(all_img_features)):
            plt.plot([features_2d[i, 0], features_2d[len(all_img_features)+i, 0]], 
                    [features_2d[i, 1], features_2d[len(all_img_features)+i, 1]], 
                    c='black', alpha=0.5, linewidth=0.2)

        from matplotlib.ticker import FormatStrFormatter
        from matplotlib.ticker import Formatter
        class DecimalFormatter(Formatter):
            def __call__(self, x, pos=None):
                # print(f"{x:.2f}".lstrip('0'))
                strs = f"{x:.2f}"
                if strs[0]=='-':
                    return strs[0]+strs[2:]
                else:
                    return strs[1:]

                return f"{x:.2f}".lstrip('0')

        plt.gca().xaxis.set_major_formatter(DecimalFormatter())
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # 添加文本标签
        plt.text(0.1,0.4, "Gap Distance: {:.2f}".format(average_distance), fontsize=25, ha='left', va='top', transform=plt.gca().transAxes)
        print(average_distance)
        # plt.grid(True, linestyle='--', alpha=0.5)

        # 显示图例
        plt.legend(loc='upper left', fontsize=20)

        
        # plt.xlabel('a',fontsize=40)
        # #设置x轴标签及其字号
        # plt.ylabel('b',fontsize=40)
        plt.tick_params(labelsize=20)

        # 保存图像
        plt.savefig("./OGE.jpg")

def plot_text(args, model, device, dataloader):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('./modelPth/bert')

    with torch.no_grad():
        model.eval()
        a_feature= []
        b_feature= []
        for step, (image, text, label) in enumerate(dataloader):
            image = image.to(device)
            label = label.to(device)
            text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(image.device)
            

            # a = model.text_encoder(spec)
            text_embeds = model.text_encoder(text_input.input_ids,
                                                 attention_mask=text_input.attention_mask,
                                                 return_dict=True
                                                 ).last_hidden_state[:,0,:]
            image_embeds = model.visual_encoder(image)

            
            # v = model.visual_net(image)

            a = model.fusion_module.t_in(text_embeds)
            v = model.fusion_module.v_in(image_embeds)


            # (_, C, H, W) = v.size()
            # B = a.size()[0]
            # v = v.view(B, -1, C, H, W)
            # v = v.permute(0, 2, 1, 3, 4)

            # a = F.adaptive_avg_pool2d(a, 1)
            # v = F.adaptive_avg_pool3d(v, 1)

            a = torch.flatten(a, 1)
            v = torch.flatten(v, 1)


            a_norm = F.normalize(a, p=2, dim =-1)
            v_norm = F.normalize(v, p=2, dim =-1)
            a_feature.append(a_norm)
            b_feature.append(v_norm)
        


        all_img_features = torch.cat(a_feature, dim=0)
        all_text_features = torch.cat(b_feature, dim=0)


        def svd(X, n_components=2):
            # using SVD to compute eigenvectors and eigenvalues
            # M = np.mean(X, axis=0)
            # X = X - M
            U, S, Vt = np.linalg.svd(X)
            # print(S)
            return U[:, :n_components] * S[:n_components]
        
        all_img_features = all_img_features[0:50,:]
        print(all_img_features.shape)
        
        all_text_features = all_text_features[0:50,:]
        print(all_text_features.shape)

        distances = []
        for i in range(len(all_img_features)):
            img_feature = all_img_features[i]
            text_feature = all_text_features[i]
            distance = torch.norm(img_feature - text_feature).item()  
            distances.append(distance)

        # 计算平均距离
        average_distance = torch.tensor(distances).mean().item() 

        features_2d = svd(torch.cat([all_img_features, all_text_features], 0).cpu().detach().numpy())
        plt.figure(figsize=(5, 5))

        # 绘制图像特征散点图
        plt.scatter(features_2d[:-len(all_img_features), 0], features_2d[:-len(all_img_features), 1], c='orange', s=40, label='Image Features', marker='o')

        # 绘制文本特征散点图
        plt.scatter(features_2d[-len(all_img_features):, 0], features_2d[-len(all_img_features):, 1], c='blue', s=40, label='Text Features', marker='x')

        # 绘制连接线
        for i in range(len(all_img_features)):
            plt.plot([features_2d[i, 0], features_2d[len(all_img_features)+i, 0]], 
                    [features_2d[i, 1], features_2d[len(all_img_features)+i, 1]], 
                    c='black', alpha=0.5, linewidth=0.2)

        # 添加文本标签
        plt.text(0.4,0.2, "Gap Distance: {:.2f}".format(average_distance), fontsize=15, ha='left', va='top', transform=plt.gca().transAxes)
        print(average_distance)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 显示图例
        plt.legend(loc=6, fontsize=12)

        # 保存图像
        plt.savefig("./Twitter_OGE.jpg", dpi=400)








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
    model = AVClassifier(args)

    

    if args.dataset == 'kinetics_sound':
        test_dataset = VADataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'Twitter15':
        test_dataset = SarcasmDataset(ann_file= "./dataset/Twitter15/annotations/test.tsv", type="test",image_root="./dataset/Twitter15/twitter2015_images")
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
    print(model)
    print('Trained model loaded!')

    plot(args, model, device, test_dataloader)
    # print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
