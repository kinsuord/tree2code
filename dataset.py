from torch.utils.data import Dataset
import os
from PIL import Image
from utils.tree import Tree
import xml.etree.ElementTree as ET

class Pix2TreeDataset(Dataset):
    def __init__(self, img_dir='../dataset/pix2code_png', img_transform=None,
                 tree_dir='../dataset/pure_tag', tree_transform=None, 
                 partition=None):
        # if partion == None find all in the floder
        if partition == None:
            self.partition = range(len([f for f in os.listdir(img_dir) 
                            if os.path.isfile(os.path.join(img_dir, f))]))
        else:
            self.partition = partition
        self.img_dir = img_dir
        self.tree_dir = tree_dir
        self.img_transform = img_transform
        self.tree_transform = tree_transform
        self.word_dict = None
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, str(idx) + '.png'))
        
        tree = ET.parse(os.path.join(self.tree_dir, str(idx) + '.xml'))
        et_root = tree.getroot()
        def build_tree(ele, et_ele):
            for et_child in et_ele:
                new_child = Tree(et_child.tag)
                ele.add_child(new_child)
                build_tree(new_child, et_child)
            ele.add_child(Tree('end'))
        root = Tree(et_root.tag)
        build_tree(root, et_root)

        if self.tree_transform != None:
            root = self.tree_transform(root)
        if self.img_transform != None:
            img = self.img_transform(img)
        return {'img': img, 'tree': root}
    
    def __len__(self):
        return len(self.partition)