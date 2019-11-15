from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import datetime
import numpy as np
import os

from utils.tree import Tree, tree_similarity
from dataset import Pix2TreeDataset
from utils.transforms import Rescale, WordEmbedding, TreeToTensor, Vec2Word
from models import Pix2TreeKai
from utils.generator import Env
from compiler import compiler


def batch_collate(batch):
    out =dict()
    out['img'] = torch.utils.data.dataloader.default_collate(
                                            [x['img'] for x in batch])
    out['tree'] = [x['tree'] for x in batch]
    return out

def train(save_name, model, train_data, pretrain=None, epoch=2, lr=1e-5, 
          batch_size=1, num_worker=2, device=torch.device("cuda:0"), 
          loss_freq=10, save_freq=700):
    
    dataloader = DataLoader(train_data, batch_size=1, pin_memory=True,
                            collate_fn=batch_collate, num_workers=num_worker)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    pretrain_e = 0
    if pretrain == None:
        # initialize
        def weights_init_uniform_rule(m):
            classname = m.__class__.__name__
            # for every Linear layer in a model..
            if classname.find('Linear') != -1:
                # get the number of the inputs
                n = m.in_features
                y = 1.0/np.sqrt(n)
                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0)
    
        model.apply(weights_init_uniform_rule)
    else:
        checkpoint = torch.load(pretrain)
        image_caption_model.load_state_dict(checkpoint['model_state_dict'])
        pretrain_e = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
    model.to(device)
    model.train()
    for e in range(pretrain_e, pretrain_e + epoch):
        for i, batch_sample in enumerate(dataloader):

            tree = batch_sample['tree']
            img = batch_sample['img']
            for t in tree:
                t.for_each_value(lambda x: x.to(device))
            img = img.to(device)
            
            # split tree to get each node
            copy_tree = tree[0].copy()
            queue = [copy_tree]
            bfs_seq = []
            
            while len(queue) != 0:
                # add new childern to queue
                node = queue.pop(0)
                bfs_seq.append(node)
                queue += node.children
            
            split_trees = []
            split_node = []
            split_parent = []
            for j in range(1,len(bfs_seq)):
                node = bfs_seq[len(bfs_seq)-j]
                split_node.append(node)
                split_parent.append(node.parent)
                node.parent.num_children -= 1
                node.parent.children.remove(node)
                split_trees.append(copy_tree.copy())
                
                l = len(split_trees)
                if l >= batch_size or j==len(bfs_seq)-1:
                    answer = torch.stack([n.value for n in split_node], dim=0)
                    parents = torch.stack([n.value for n in split_parent], dim=0)
                    imgs = torch.cat([img]*l, dim=0)
                    pred = image_caption_model(imgs, split_trees, parents)
                    split_trees = []
                    split_node = []
                    split_parent = []
                    
                    optimizer.zero_grad()
                    loss = criterion(pred, answer)
                    loss.backward()
                    optimizer.step()

            if i%loss_freq == 0:
                print('epoch:{} tree:{} loss:{}'.format(
                        e, i, loss))

            if i%save_freq == 0:
                checkpoint_path = os.path.join(
                        'checkpoint', '{}_{}_{}.pth'.format(save_name, e, i))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': image_caption_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss':  loss
                }, checkpoint_path)
                print(datetime.datetime.now(), 'save model to {}'.
                                                     format(checkpoint_path))
        
        checkpoint_path = os.path.join(
                'checkpoint', '{}_{}.pth'.format(save_name, e))
        torch.save({
            'epoch': e,
            'model_state_dict': image_caption_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, checkpoint_path)
        print(datetime.datetime.now(), 'save model to {}'.
                                              format(checkpoint_path))
    return model

def valid(valid_data, model, word_dict, device=torch.device("cuda:0"),
          load_predict=None, save_predict=None, with_rule=True):
    scores = []
    model.eval()
    env = Env(rule='rule_c.json')
    
    if load_predict!=None:
        preds = np.load(load_predict, allow_pickle=True).item()
    else:
        preds = []        

    for i in range(len(valid_data)):
        img = valid_data[i]['img'].to(device).unsqueeze(0)
        
        if load_predict != None:
            pred = preds[i]
        else:
            if with_rule:
                pred = predict_tree_with_rule(img, model, device, word_dict, env)
            else:
                pred = predict_tree(img, model, device, word_dict)
            preds.append(pred)
        score = tree_similarity(pred, valid_data[i]['tree'])
        scores.append(score)

    if save_predict != None:
        np.save(save_predict, preds)

    print('Average score: {}'.format(sum(scores)/len(scores)))
    return scores
        

def predict_tree(img, model, device, word_dict, max_child=6):
    tranf = transforms.Compose([WordEmbedding(word_dict), TreeToTensor()])
    end_value = tranf(Tree('end')).value.to(device)
    model.to(device)
    
    root = Tree('root')   
    root = tranf(root)
    root.value = root.value.to(device)
    out_size = root.value.size()
    
    queue = [root]
    while len(queue) != 0:
        sub_tree = Tree(image_caption_model(img, [root]).flatten().detach())
        max_value = torch.max(sub_tree.value)
        sub_tree.value = torch.where(sub_tree.value >= max_value, 
                torch.ones(out_size).to(device),
                torch.zeros(out_size).to(device))
        queue[0].add_child(sub_tree)
        
        if len(queue[0].children) >= max_child:
            sub_tree = Tree(end_value.clone().detach())
            queue[0].add_child(sub_tree)
        if torch.equal(end_value, sub_tree.value):
            queue.pop(0)
        else:
           queue.append(sub_tree)
           
    root.for_each_value(lambda x: x.cpu().numpy())
    vec2word = Vec2Word(word_dict)
    root = vec2word(root)
    return root

def predict_tree_with_rule(img, model, device, word_dict, env):
    env.reset()
    vec_dict = dict()
    for k, v in word_dict.items():
        vec_dict[np.sum(np.multiply(v, np.arange(v.size)))] = k
    tranf = transforms.Compose([WordEmbedding(word_dict), TreeToTensor()])
    model.to(device)
    root, parent, chioce = env.state()
    out_size = len(word_dict)
    while parent != None:
        # get mask
        mask = np.sum([word_dict[c] for c in chioce], axis=0)
        copy_root = root.copy()
        root_tensor = tranf(copy_root).for_each_value(lambda x: x.to(device))
        parent_tensor = torch.from_numpy(word_dict[parent.value]).to(device).float()
        parent_tensor = torch.stack([parent_tensor])
        pred_node = image_caption_model(img, [root_tensor], parent_tensor).flatten().detach()

        # fliter the new node and get the predict value
        pred_node *= torch.from_numpy(mask).to(device).float()
        max_value = torch.max(pred_node)
        pred_node = torch.where(pred_node >= max_value, 
                torch.ones(out_size).to(device),
                torch.zeros(out_size).to(device))

        action = vec_dict[np.sum(
                np.multiply(pred_node.cpu().numpy(), np.arange(out_size)))]

        root, parent, chioce = env.step(action)
    return root

if __name__ == '__main__': 
    # some arguments
    dataset_tree_dir = './dataset/tree'
    dataset_img_dir = './dataset/img'

    # load word dict
    def count_word_dict(dataset):
        word_count = {'root':0, 'end':0}
        def count_tree(tree, word_count):
            for child in tree.children:
                count_tree(child, word_count)
            if tree.value in word_count:
                word_count[tree.value] += 1
            else:
                word_count[tree.value] = 1
        
        for i in range(len(dataset)):
            count_tree(dataset[i]['tree'], word_count)
        
        word_dict = {}
        i = 0
        for key in word_count.keys():
            a = np.zeros(len(word_count))
            a[i] = 1.0
            word_dict[key] = a
            i += 1
        return word_dict

    dataset = Pix2TreeDataset(
            img_dir=dataset_img_dir, tree_dir=dataset_tree_dir)
    if not os.path.exists('word_dict.npy'):
        word_dict = count_word_dict(dataset)
        np.save('word_dict.npy', word_dict)
    else:
        word_dict = np.load('word_dict.npy', allow_pickle=True).item()
        
    # prepare dataset
    train_data = Pix2TreeDataset(
            img_dir=dataset_img_dir, tree_dir=dataset_tree_dir,
            partition=range(int(len(dataset)*0.8)),
            tree_transform=transforms.Compose([WordEmbedding(word_dict),
                                               TreeToTensor()]),
            img_transform=transforms.Compose([Rescale(224),
                                              transforms.ToTensor()]))

    valid_data = Pix2TreeDataset(
            img_dir=dataset_img_dir, tree_dir=dataset_tree_dir,
            partition=range(int(len(dataset)*0.8), len(dataset)),
            img_transform=transforms.Compose([Rescale(224),
                                              transforms.ToTensor()]))
    #import matplotlib.pyplot as plt
    #print(dataset[0]['tree'])
    #plt.imshow(dataset[0]['img'])
 
    # model
    # image_caption_model = Pix2TreeKai(len(word_dict))
    
    # train('Kai', image_caption_model, train_data, epoch=2, batch_size=10, 
    #         lr = 10e-6)
    
################################## test model #################################
    checkpoint = torch.load("checkpoint/Kai_1.pth")
    image_caption_model = Pix2TreeKai(len(word_dict))
    image_caption_model.load_state_dict(checkpoint['model_state_dict'])
    scores = valid(valid_data, image_caption_model, word_dict, save_predict='predXrule.npy', with_rule=True)