# -*- coding: utf-8 -*-
import utils
import os
from shutil import copyfile

#%% train test split
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