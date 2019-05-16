# -*- coding: utf-8 -*-
from tree import Tree


class Env():
    def __init__(self):
        self.reset()
    
    def state(self):
        pass
        
    def step(self, action):
        return done, reward
    
    def reset(self):
        self.tree = Tree('root')