from torchvision import transforms
from .tree import Tree
import numpy as np
import torch

class Rescale(object):
    def __init__(self, output_size=224):
        self.tsfrm = transforms.Resize((output_size, output_size))

    def __call__(self, sample):
        sample = self.tsfrm(sample)
        return sample.convert('RGB')

class WordEmbedding(object):
    def __init__(self, word_dict):
        self.word_dict = word_dict
    
    def __call__(self, sample): 
        sample.for_each_value(lambda x: self.word_dict[x])
        return sample
    
class Vec2Word(object):
    def __init__(self, word_dict):
        self.vec_dict = dict()
        for k, v in word_dict.items():
            self.vec_dict[np.sum(np.multiply(v, np.arange(v.size)))] = k
        
    
    def __call__(self, sample):
        sample.for_each_value(lambda x: self.vec_dict[
                np.sum(np.multiply(x, np.arange(x.size)))])
        return sample

class TreeToTensor(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        sample.for_each_value(lambda x: torch.from_numpy(x).float())
        return sample
    
class TreePadding(object):
    def __init__(self, word_dict, pad_value='pad', size=[5, 5, 4, 1]):
        self.size = size
        self.pad_value = pad_value

    def __call__(self, sample, ):
        queue = [sample]
        while len(queue) != 0:
            max_children = self.size[queue[0].height()]
            if len(queue[0].children) < max_children:
                for i in range(len(queue[0].children), max_children):
                    queue[0].add_child(Tree(self.pad_value))
            queue.pop(0)
            for child in queue[0].children:
                queue.append(child)