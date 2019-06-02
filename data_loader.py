import os
import json
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image

from utils import json2tree, vec2word, dsl2tree

class TreeDataset(Dataset):
    def __init__(self, tree_dir='dataset/pix2code/dsl', embedding=True, dsl=True, device='cpu'):
        self.device = device
        files = [f for f in os.listdir(tree_dir) if os.path.isfile(os.path.join(tree_dir, f))]
        if dsl:
            dsl_data = []
            for file in files:    
                with open(os.path.join(tree_dir, file)) as dsl_file: 
                    dsl_data.append(dsl_file.read())
                    
            self.trees = [dsl2tree(t) for t in dsl_data]
        else:
            json_data = []
            for file in files:    
                with open(os.path.join(tree_dir, file)) as json_file: 
                    json_data.append(json.load(json_file))
                    
            self.trees = [json2tree(t) for t in json_data]
        
        word_count = {}
        def count_tree(tree, word_count):
            for child in tree.children:
                count_tree(child, word_count)
            if tree.value in word_count:
                word_count[tree.value] += 1
            else:
                word_count[tree.value] = 1
        
        for tree in self.trees:    
            count_tree(tree, word_count)
        
        self.word_dict = {}
        
        i = 0
        for key in word_count.keys():
            a = np.zeros(len(word_count))
            a[i] = 1.0
            self.word_dict[key] = a
            i += 1
        if embedding:
            for tree in self.trees:    
                self._word_embedding(tree)
        
    def __getitem__(self, idx):
        queue = [self.trees[idx]]
        while len(queue) != 0:
            node = queue.pop(0)
            node.value = torch.tensor(node.value).to(self.device).float()
            queue += node.children
        return self.trees[idx]
    
    def __len__(self):
        return len(self.trees)
    
    def __str__(self):
        out_str = 'average tree size {:.2f}'.format(sum([tree.size() for tree in self.trees])/len(self.trees))
        out_str += '\n' + 'average tree depth {}'.format(sum([tree.depth() for tree in self.trees])/len(self.trees))
        return out_str
    
    def _word_embedding(self, tree):
        tree.value = self.word_dict[tree.value]
        for child in tree.children:
            self._word_embedding(child)
        return tree
    
class ImgDataset(Dataset):
    def __init__(self, img_dir='dataset/pix2code/png', transform=transforms.Compose([transforms.Resize((224, 224))]), device='cpu'):
        self.transform = transform
        self.device = device
        self.img_path = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                        if os.path.isfile(os.path.join(img_dir, f))]
    
    def __getitem__(self, idx):
        img=Image.open(self.img_path[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return torch.tensor(np.array(img).reshape(-1, 224, 224, 3).transpose(0, 3, 1, 2)).to(self.device).float()
    
    def __len__(self):
        return len(self.img_path)
    
class TextLoader(Dataset):
    def __init__(self, dsl_dir='dataset/pix2code/dsl', device='cpu'):
        self.device = device
        self.files = [f for f in os.listdir(dsl_dir) if os.path.isfile(os.path.join(dsl_dir, f))]
        
    def __getitem__(self, idx):
        pass