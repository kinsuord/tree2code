#%%
import os
import torch

#%% load data
device = torch.device("cuda:0")

from data_loader import TreeDataset, ImgDataset

tree_train = TreeDataset(tree_dir=os.path.join('bin', 'tree_train'), device=torch.device)
img_train = ImgDataset(img_dir=os.path.join('bin', 'img_train'), device=device)

#%% init model
import time
import numpy as np
from models import ShowAndTellTree
import torch.optim as optim

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

image_caption_model = ShowAndTellTree(len(tree_train.word_dict)).to(device)
image_caption_model.apply(weights_init_uniform_rule)

#%% training
from tensorboardX import SummaryWriter

writer = SummaryWriter('tensorboard/exp-1')

epoch = 1
lr = 1e-5
name = 'sdlTree'
print_frequency = 1
save_frequency = 100
optimizer = optim.Adam(image_caption_model.parameters(), lr=lr)
criterion = torch.nn.BCELoss()
image_caption_model.train()

for e in range(epoch):
    start = time.time()
    for i , (tree, img) in enumerate(zip(tree_train, img_train)):
        # genarate new dataset
        split_tree = tree.copy()
        queue = [split_tree]
        bfs_seq = []
        while len(queue) != 0:
            # add new childern to queue
            node = queue.pop(0)
            bfs_seq.append(node)
            queue += node.children
        losses = []
        while len(bfs_seq) > 1:
            # get tree and next node
            node = bfs_seq.pop()
            node.parent.num_children -= 1
            node.parent.children.remove(node)
            next_node = image_caption_model(img, split_tree)
            
            # training
            optimizer.zero_grad()
            loss = criterion(next_node, node.value.view(1,-1))
            writer.add_scalar('loss', loss, i)
            losses.append(loss)  
            loss.backward()
            optimizer.step()
                        
        if i%print_frequency == 0:
            print('epoch:{} tree:{} loss:{}'.format(e, i, sum(losses)/len(losses)))
            
        if i%save_frequency == 0:
            checkpoint_path = os.path.join(
                    'checkpoint', '{}_{}_{}.pth'.format(name, e, i))
            torch.save({
                'epoch': epoch,
                'model_state_dict': image_caption_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':  sum(losses)/len(losses)
            }, checkpoint_path)
            print('save model to checkpoint_path')
            
    end = time.time()
    print('epoch: {} time: {:2f}'.format(e, end-start))
    
#%%  evaluate

from tree import Tree

def predictTree(image_caption_model, img, word_dict):
    
    device = torch.device("cuda:0")

    root = Tree(word_dict["root"])
    root.value = torch.tensor(root.value).to(device).float()
    while(1):
        sub_tree = Tree(image_caption_model(img, root).flatten())
        max = 0
        max_index = 0

        for i in range(14):
            if sub_tree.value[i]>max:
                max = sub_tree.value[i]
                max_index = i
        for i in range(14):
            if i != max_index:
                sub_tree.value[i] = 0;
            else:
                sub_tree.value[i] = 1;
        root.add_child(sub_tree)
        if sub_tree.value[2] == 1:
            break
    for child in root.children:
        if child.value[3] == 1 or child.value[7] == 1 or child.value[8] == 1 or child.value[9] == 1 or child.value[12] == 1:
            predictSubTree(image_caption_model, img, child, root, 1, word_dict)
    return root

def predictSubTree(image_caption_model, img, now_node, root, depth, word_dict):
    if depth>3:
        return
    device = torch.device("cuda:0")

    count = 0
    while(1):
        count = count +1
        if count == 10:
            end_node = Tree(word_dict["None"])
            end_node.value = torch.tensor(end_node.value).to(device).float()
            now_node.add_child(end_node)
            return
        
        sub_tree = Tree(image_caption_model(img, root).flatten())
        max = 0
        max_index = 0

        for i in range(14):
            if sub_tree.value[i]>max:
                max = sub_tree.value[i]
                max_index = i
        for i in range(14):
            if i != max_index:
                sub_tree.value[i] = 0
            else:
                sub_tree.value[i] = 1
        now_node.add_child(sub_tree)
        if sub_tree.value[2] == 1:
            break
    for child in now_node.children:
        if child.value[3] == 1 or child.value[7] == 1 or child.value[8] == 1 or child.value[9] == 1 or child.value[12] == 1:
            predictSubTree(image_caption_model, img, child, root, depth+1, word_dict)

#%%
checkpoint = torch.load("checkpoint/sdlTree_0_1300.pth")
image_caption_model.load_state_dict(checkpoint['model_state_dict'])
tree_eval = TreeDataset(tree_dir=os.path.join('bin', 'tree_eval'), device=device)
img_eval = ImgDataset(img_dir=os.path.join('bin', 'img_eval'), device=device)

#%%
pred = predictTree(image_caption_model, img_eval[0].to(device).float(), tree_train.word_dict)

#%%
def to_cpu(x):
    return x.to(torch.device('cpu'))

pred.for_each_value(lambda x: x.to(torch.device('cpu')))