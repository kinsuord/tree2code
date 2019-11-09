# -*- coding: utf-8 -*-
import torch
import numpy as np
from torchvision import transforms

from dataset import Pix2TreeDataset
from utils.transforms import Rescale, WordEmbedding, TreeToTensor, Vec2Word

def train(
          name,
          train_dataset,
          g_model, q_model,
          pretrain_step,
          bucket_size, batch_size,
          global_step, d_step, g_step,
          d_lr, g_lr,
          steps_per_log, steps_per_checkpoint,
          checkpoint_dir, summery_file
          ):
    # 
    for e in range(global_step):
        for _ in d_step:
            pass
        for _ in g_step:
            pass
        

if __name__ == '__main__':
    word_dict = np.load('word_dict.npy', allow_pickle=True).item()
    dataset = Pix2TreeDataset()
    train_data = Pix2TreeDataset(partition=range(int(len(dataset)*0.8)),
            tree_transform=transforms.Compose([WordEmbedding(word_dict),
                                               TreeToTensor()]),
            img_transform=transforms.Compose([Rescale(224),
                                              transforms.ToTensor()]))

    G_pretrain = torch.load("checkpoint/batch10_2_2.pth")

#    pred = np.load("pred.npy", allow_pickle=True)
#    predXrule = np.load("predXrule.npy", allow_pickle=True)
#    
#    l = [p.size() for p in predXrule]
#    print(sum(l)/len(l))
