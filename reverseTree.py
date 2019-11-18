import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import time;

from dataset import Pix2TreeDataset
from utils import transforms as trsf
from utils.generator import Env
from utils.tree import Tree, tree_similarity
from models import Pix2TreeReverse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # some arguments
    dataset_tree_dir = './dataset/xml'
    dataset_img_dir = './dataset/img'
    train_valid_ratio = 0.8

    dataset = Pix2TreeDataset(
            img_dir=dataset_img_dir, tree_dir=dataset_tree_dir
    )
    word_dict = get_word_dict('word_dict.npy', dataset)

    # prepare dataset
    train_data = Pix2TreeDataset(
            img_dir=dataset_img_dir, tree_dir=dataset_tree_dir,
            partition=range(int(len(dataset)*train_valid_ratio)),
            tree_transform=transforms.Compose([trsf.WordEmbedding(word_dict), trsf.TreeToTensor()]),
            img_transform=transforms.Compose([trsf.Rescale(224), transforms.ToTensor()]))

    valid_data = Pix2TreeDataset(
            img_dir=dataset_img_dir, tree_dir=dataset_tree_dir,
            partition=range(int(len(dataset)*train_valid_ratio), len(dataset)),
            # tree_transform=transforms.Compose([trsf.WordEmbedding(word_dict), trsf.TreeToTensor()]),
            img_transform=transforms.Compose([trsf.Rescale(224),transforms.ToTensor()]))

    model = Pix2TreeReverse(len(word_dict), 1024)
    # train('reverse', model, train_data, valid_data, word_dict)
    checkpoint = torch.load('checkpoint/reverse_0.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    valid(model, valid_data, word_dict,valid_num=30, show_result=False)

def batch_collate(batch):
    out =dict()
    out['img'] = torch.utils.data.dataloader.default_collate([x['img'] for x in batch])
    out['tree'] = [x['tree'] for x in batch]
    return out

def train(name, model, train_data, valid_data, word_dict, 
            checkpoint='', epoch=2, 
            learning_rate=1e-5, batch_size=1, num_workers=1,
            tree_per_log=1, epoch_per_save=1):

    dataloader = DataLoader(train_data, batch_size=1, pin_memory=True, shuffle=True, 
                            collate_fn=batch_collate, num_workers=num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    if checkpoint == '':
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

    model.to(device)
    model.train()

    tree_count = 0
    losses = AverageMeter()

    for e in range(epoch):
        for i, batch in enumerate(dataloader):
            tree = batch['tree'][0]
            img = batch['img']
            tree.for_each_value(lambda x: x.to(device))
            img = img.to(device)

            # find all leaf
            leaf = []
            queue = [tree]
            while len(queue) != 0:
                node = queue.pop(0)
                if len(node.children) == 0:
                    leaf.append(node)
                queue += node.children

            for l in leaf:
                # get the path to parent
                path = []
                n = l.parent
                path.insert(0, n)
                while n!=tree:
                    n = n.parent
                    path.insert(0, n)
                pred = model(img, path)
                path.pop(0)
                path.append(l)
                target = [node.value for node in path]
                target = torch.stack(target, dim=0)

                optimizer.zero_grad()
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()

                losses.update(loss, target.size()[0])
            tree_count += 1

            if tree_count % tree_per_log == 0:
                time_str = time.strftime("%H:%M:%S", time.localtime())
                print('[{}] epoch: {} tree: {} loss: {}'.format(time_str, e, i, losses.avg))
                losses.reset()

        if e % epoch_per_save == 0:
            checkpoint_path = 'checkpoint/{}_{}.pth'.format(name, e)
            print('save checkpoint: {}'.format(checkpoint_path))
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

            valid(valid_data[:2], show_result=True, rule='rule.json')
                

    return model

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load(pth_file):
    pass

def valid(model, valid_data, word_dict, valid_num=None, show_result=False, rule=''):
    if rule != '':
        env = Env(rule=rule)
    else:
        env = None
    if valid_num==None:
        valid_num = len(valid_data)

    scores = AverageMeter()
    model.to(device)
    model.eval()

    for i in range(valid_num):
        img = valid_data[i]['img'].to(device).unsqueeze(0)
        if rule == '':
            pred = predict_tree(model, img, word_dict, valid_data[i]['tree'])
        else:
            pred = predict_tree_with_rule(model, img, word_dict, env)
            torch.cuda.empty_cache()

        score = tree_similarity(pred, valid_data[i]['tree'])
        if show_result:
            print('==========Target: {}========='.format(valid_data.get_path(i)))
            print(valid_data[i]['tree'])
            print('==========Predict=========')
            print(pred)
            print('==========Score: {}========='.format(score))
            print()
        scores.update(score)

    print('[Valid] Average score:{}'.format(scores.avg))

    #############################
    # losses = AverageMeter()
    # criterion = torch.nn.BCELoss()

    # tree = valid_data[0]['tree']
    # img = valid_data[0]['img']
    # tree.for_each_value(lambda x: x.to(device))
    # img = img.to(device).unsqueeze(0)

    # # find all leaf
    # leaf = []
    # queue = [tree]
    # while len(queue) != 0:
    #     node = queue.pop(0)
    #     if len(node.children) == 0:
    #         leaf.append(node)
    #     queue += node.children

    # for l in leaf:
    #     # get the path to parent
    #     path = []
    #     n = l.parent
    #     path.insert(0, n)
    #     while n!=tree:
    #         n = n.parent
    #         path.insert(0, n)
    #     pred = model(img, path)
    #     path.pop(0)
    #     path.append(l)
    #     target = [node.value for node in path]
    #     target = torch.stack(target, dim=0)

    #     loss = criterion(pred, target)

    #     print(pred[-1])
    #     losses.update(loss, target.size()[0])

    # import pdb; pdb.set_trace()

def predict_tree(model, img, word_dict, tree, max_child=7):

    def str_to_tensor(word):
        return torch.from_numpy(word_dict[word]).float().to(device)

    def get_path(node, root):
        path = []
        n = node
        path.insert(0, n)
        while n!=root:
            n = n.parent
            path.insert(0, n)
        return path
    
    def tree_str(node):
        out_str = '  '*node.height() + node.str
        for child in node.children:
            child_str = tree_str(child)
            out_str += '\n' + child_str
        return out_str

    root = Tree(str_to_tensor('root'))
    root.str = 'root'

    target_queue = [tree]
    target_index = 0

    queue = [root]
    while len(queue) != 0:
        parent = queue[0]
        path = get_path(parent, root)

        pred = model(img, path).detach()
        pred_value = pred[-1]
        pred_node = Tree(pred_value)
        word = list(word_dict.keys())[torch.argmax(pred_node.value)]

        # check ans
        target_parent = target_queue[0]
        
        if word!=target_parent.children[target_index].value:
            print('wrong!!! target:{} predict:{}'.format(target_parent.children[target_index].value, word))
            # print(tree_str(root))
            # print("====================")
            word = target_parent.children[target_index].value

        pred_node.str = word


        parent.add_child(pred_node)
        if pred_node.str == 'end':
            target_index = 0
            target_queue.pop(0)
            queue.pop(0)
        else:
            target_queue.append(target_parent.children[target_index])
            queue.append(pred_node)
            target_index += 1
    # tranf = transforms.Compose([WordEmbedding(word_dict), TreeToTensor()])
    # end_value = tranf(Tree('end')).value.to(device)
    # model.to(device)
    
    # root = Tree('root')   
    # root = tranf(root)
    # root.value = root.value.to(device)
    # out_size = root.value.size()
    
    # queue = [root]
    # while len(queue) != 0:
    #     sub_tree = Tree(image_caption_model(img, [root]).flatten().detach())
    #     max_value = torch.max(sub_tree.value)
    #     sub_tree.value = torch.where(sub_tree.value >= max_value, 
    #             torch.ones(out_size).to(device),
    #             torch.zeros(out_size).to(device))
    #     queue[0].add_child(sub_tree)
        
    #     if len(queue[0].children) >= max_child:
    #         sub_tree = Tree(end_value.clone().detach())
    #         queue[0].add_child(sub_tree)
    #     if torch.equal(end_value, sub_tree.value):
    #         queue.pop(0)
    #     else:
    #        queue.append(sub_tree)
           
    # root.for_each_value(lambda x: x.cpu().numpy())
    # vec2word = Vec2Word(word_dict)
    # root = vec2word(root)
    # return root

def predict_tree_with_rule(model, img, word_dict, env):
    pass
    # env.reset()
    # root, parent, chioce = env.state()
    # while parent != None:

    # env.reset()
    # vec_dict = dict()
    # for k, v in word_dict.items():
    #     vec_dict[np.sum(np.multiply(v, np.arange(v.size)))] = k
    # tree_to_tensor = transforms.Compose([trsf.WordEmbedding(word_dict), trsf.TreeToTensor()])

    # root, parent, chioce = env.state()
    # out_size = len(word_dict)
    # while parent != None:

    #     path = []
    #     n = parent
    #     path.insert(0, n)
    #     while n!=root:
    #         n = n.parent
    #         path.insert(0, n)

    #     copy_root = root.copy()
    #     copy_path = [copy_root]
    #     n = copy_root
    #     for i in range(1, len(path)):
    #         idx = path[i-1].children.index(path[i])
    #         n = n.children[idx]
    #         copy_path.append(n)

    #     path_str = [node.value for node in path]
    #     # print(' -> '.join(path_str))
    #     tree_to_tensor(copy_root).for_each_value(lambda x: x.to(device))
    #     pred_node = model(img, copy_path).detach()[-1].clone().detach()
    #     # import pdb; pdb.set_trace()
    #     # pred_node = pred_node
        
    #     # print('adssa')
    #     # import pdb; pdb.set_trace()
    #     # parent_tensor = torch.from_numpy(word_dict[parent.value]).to(device).float()
    #     # parent_tensor = torch.stack([parent_tensor])
    #     # pred_node = model(img, [root_tensor], parent_tensor).flatten().detach()

    #     # fliter the new node and get the predict value
    #     mask = np.sum([word_dict[c] for c in chioce], axis=0)
    #     pred_node *= torch.from_numpy(mask).to(device).float()
    #     max_value = torch.max(pred_node)
    #     pred_node = torch.where(pred_node >= max_value, 
    #             torch.ones(out_size).to(device),
    #             torch.zeros(out_size).to(device))

    #     action = vec_dict[np.sum(
    #             np.multiply(pred_node.cpu().numpy(), np.arange(out_size)))]
    #     # print(action)

    #     root, parent, chioce = env.step(action)
    # return root

def get_word_dict(dict_file, dataset):

    # get word_dict from dict_file, if not count dict form dataset and save to file
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

    if not os.path.exists(dict_file):
        word_dict = count_word_dict(dataset)
        np.save(dict_file, word_dict)
    else:
        word_dict = np.load(dict_file, allow_pickle=True).item()

    return word_dict

if __name__ == '__main__':
    main()
