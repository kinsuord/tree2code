# -*- coding: utf-8 -*-

import torch
import torch.nn as nn 
import torchvision.models as models

class ChildSumTreeLSTM(nn.Module):
    '''
    input: Tree(in_dim)
    output: state(mem_dim)
    '''
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx])

        inputs = tree.value
        if tree.num_children == 0:
            child_c = inputs.detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs.detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs, child_c, child_h)
        return tree.state

class BatchModel(nn.Module):
    '''
    input: img(batch_size, 224, 224, 3), List(Tree())(length: batch_size)
    output: pred(batch_size, word_dim)
    '''
    def __init__(self, word_dim):
        super(BatchModel, self).__init__()
        self.word_dim = word_dim
        
        self.tree_lstm = ChildSumTreeLSTM(word_dim, 512)
        self.cnn = models.vgg11_bn(pretrained=True) # 1000
        self.fc = nn.Sequential(
            nn.Linear(1512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, word_dim),
            nn.Softmax()
        )
        
    def forward(self, img, tree):
#        import pdb; pdb.set_trace()
        img_features = self.cnn(img)
        tree_features = []
        for i in range(len(tree)):
            tree_features.append(self.tree_lstm(tree[i])[0])
        tree_features = torch.cat(tree_features, dim=0)
        import pdb; pdb.set_trace()
#        img_features = torch.cat([img_feature] * (len(tree)-1), dim=0)
#        tree_vectors = []
#        for i in range(1,len(tree)):
#            node = tree.pop()
#            node.parent.num_children -= 1
#            node.parent.children.remove(node)
#            tree_vectors.append(self.tree_lstm(tree[0])[0])
#        tree_states = torch.cat(tree_vectors, dim=0)
        return self.fc(torch.cat((img_features, tree_features), dim=1))
    
  
class ShowAndTellTree(nn.Module):
    '''
    input: img(224, 224, 3), Tree(word_dim)
    output: next_word(word_dim)
    '''
    def __init__(self, word_dim):
        super(ShowAndTellTree, self).__init__()
        self.word_dim = word_dim
        
        self.tree_lstm = ChildSumTreeLSTM(word_dim, 1024)
        self.cnn = models.vgg16(pretrained=True) # 1000
        self.fc = nn.Sequential(
            nn.Linear(2024, 2024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, word_dim),
            nn.Softmax()
        )
        
    def forward(self, img, tree):
        img_features = self.cnn(img)
        tree_state, _ = self.tree_lstm(tree)
        return self.fc(torch.cat((img_features, tree_state), dim=1))
    
class LightNNShowAndTellTree(nn.Module):
    '''
    input: img(224, 224, 3), Tree(word_dim)
    output: next_word(word_dim)
    '''
    def __init__(self, word_dim):
        super(LightNNShowAndTellTree, self).__init__()
        self.word_dim = word_dim
        
        self.tree_lstm = ChildSumTreeLSTM(word_dim, 512)
        self.cnn = models.vgg11_bn(pretrained=True) # 1000
        self.fc = nn.Sequential(
            nn.Linear(1512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, word_dim),
            nn.Softmax()
        )
        
    def forward(self, img, tree):
        img_features = self.cnn(img)
        tree_state, _ = self.tree_lstm(tree)
        return self.fc(torch.cat((img_features, tree_state), dim=1))
    
#class PaddingShowAndTellTree(nn.Module):
#     def __init__(self, word_dim, tree_size):
#       input: img(batch_size, 3, 224, 224), 
#              Tree(batch size, [5, 5, 4, 1] * word_dim)
##       output: next_word(word_dim)
#         pass
#     
#    def forward(self, img, tree):
#        pass
    
class InceptionShowAndTellTree(nn.Module):
   '''
   input: img(3, 299, 299), Tree(word_dim)
   output: next_word(word_dim)
   '''
   def __init__(self, word_dim):
       super(InceptionShowAndTellTree, self).__init__()
       self.word_dim = word_dim
       
       self.tree_lstm = ChildSumTreeLSTM(word_dim, 512)
       self.cnn = models.inception_v3(pretrained=True) # 1000
       self.fc = nn.Sequential(
           nn.Linear(2024, 1024),
           nn.ReLU(True),
           nn.Dropout(),
           nn.Linear(1024, word_dim),
           nn.Softmax()
       )
       
   def forward(self, img, tree):
       print(img.size())
       img_features = self.cnn(img)
       tree_state, _ = self.tree_lstm(tree)
       return self.fc(torch.cat((img_features, tree_state), dim=1))