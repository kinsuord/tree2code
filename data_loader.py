import os
import json
import torchvision.models as models
from torch.utils.data.dataset import Dataset

from utils import json2tree

class TreeLoader(Dataset):
    def __init__(self, json_dir='dataset/pix2code/xml_hast'):
        files = [f for f in os.listdir(json_dir) if os.path.isfile(os.path.join(json_dir, f))]
        json_data = []
        for file in files:    
            with open(os.path.join(json_dir, file)) as json_file: 
                json_data.append(json.load(json_file))
                
        self.trees = [json2tree(t) for t in json_data]
    
    def __getitem__(self, idx):
        return self.trees[idx]
    
    def __len__(self):
        return len(self.trees)
    
    def __str__(self):
        pass
    
class ImgLoader(Dataset):
    def __init__(self, img_dir='dataset/pix2code/png'):
        files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    
    def __getitem__(self, idx):
        return self.trees[idx]
    
    def __len__(self):
        return len(self.trees)
    
    def __str__(self):
        pass
    
class TextLoader(Dataset):
    def __init__(self, dsl_dir='dataset/pix2code/dsl'):
        pass
    
loader = TreeLoader()
