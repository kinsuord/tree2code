import os
import json
from torchvision import transforms
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image

from utils import json2tree, vec2word

class TreeLoader(Dataset):
    def __init__(self, json_dir='dataset/pix2code/xml_hast', embedding=True):
        files = [f for f in os.listdir(json_dir) if os.path.isfile(os.path.join(json_dir, f))]
        json_data = []
        for file in files:    
            with open(os.path.join(json_dir, file)) as json_file: 
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
    
class ImgLoader(Dataset):
    def __init__(self, img_dir='dataset/pix2code/png', transform=transforms.Compose([transforms.Resize((224, 224))])):
        self.transform = transform
        self.img_path = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                        if os.path.isfile(os.path.join(img_dir, f))]
    
    def __getitem__(self, idx):
        img=Image.open(self.img_path[idx])
        if self.transform:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.img_path)
    
class TextLoader(Dataset):
    def __init__(self, dsl_dir='dataset/pix2code/dsl'):
        pass
treeloader = TreeLoader()
