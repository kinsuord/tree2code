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

#%%
from data_loader import TreeDataset, ImgDataset

tree_train = TreeDataset(tree_dir=os.path.join('bin', 'tree_train'))
img_train = ImgDataset(img_dir=os.path.join('bin', 'img_train'))

#%%
import time
from tree import Tree
from models import ImageCaptionSimple

print_frequency = 100

def train(trees, imgs, epoch=10):
    for e in range(epoch):
        start = time.time()
        for i , (tree, img) in enumerate(zip(trees, imgs)):
            stack = [tree]
            pre_tree = Tree(trees.word_dict['root'])
            while len(stack) != 0:
                node = stack.pop()
                for children in node.children[::-1]:
                    stack.append(children)
                
            if i%print_frequency == 0:
                print('tree:{} loss:{}'.format(i, 0))
        end = time.time()
        print('epoch: {} time: {:2f}'.format(e, end-start))

train(tree_train, img_train, epoch=1)