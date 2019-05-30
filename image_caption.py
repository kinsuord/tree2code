#%% split data
import os
import utils
from shutil import copyfile

tree_dir='dataset/pix2code/dsl'
tree_files = [f for f in os.listdir(tree_dir) if os.path.isfile(os.path.join(tree_dir, f))]

img_dir='dataset/pix2code/png'
img_files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]


train_test_split_rate = 0.8
utils.createFolder(os.path.join('bin', 'tree_train'))
for file in tree_files[:int(len(tree_files)*0.8)]:    
    copyfile(os.path.join(tree_dir, file), os.path.join('bin', 'tree_train', file))
utils.createFolder(os.path.join('bin', 'tree_eval'))
for file in tree_files[int(len(tree_files)*0.8):]:    
    copyfile(os.path.join(tree_dir, file), os.path.join('bin', 'tree_eval', file))

utils.createFolder(os.path.join('bin', 'img_train'))
for file in img_files[:int(len(img_files)*0.8)]:    
    copyfile(os.path.join(img_dir, file), os.path.join('bin', 'img_train', file))
utils.createFolder(os.path.join('bin', 'img_eval'))
for file in img_files[int(len(tree_files)*0.8):]:    
    copyfile(os.path.join(img_dir, file), os.path.join('bin', 'img_eval', file))

#%% load data
from data_loader import TreeDataset, ImgDataset

tree_train = TreeDataset(tree_dir=os.path.join('bin', 'tree_train'))
img_train = ImgDataset(img_dir=os.path.join('bin', 'img_train'))

#%% train
import time
from tree import Tree
from models import ImageCaptionTree
import numpy as np

print_frequency = 100

def train(trees, imgs, epoch=10):
    image_caption_model = ImageCaptionTree(len(tree_train.word_dict))
    for e in range(epoch):
        start = time.time()
        for i , (tree, img) in enumerate(zip(trees, imgs)):
            queue = [tree]
            bfs_seq = []
            while len(queue) != 0:
                # add new childern to queue
                node = queue.pop(0)
                bfs_seq.append(node)
                queue += node.children
            while len(bfs_seq) != 0:
                node = bfs_seq.pop()
                node.parent.children.remove(node)
                next_node = image_caption_model(img, tree)
                import pdb; pdb.set_trace()
            
                            
            if i%print_frequency == 0:
                print('tree:{} loss:{}'.format(i, 0))
        end = time.time()
        print('epoch: {} time: {:2f}'.format(e, end-start))

train(tree_train, img_train, epoch=1)