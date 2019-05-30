# -*- coding: utf-8 -*-
from tree import Tree
# DSL_TYPE = {
#   root:['root'],
#   element:['header', 'row']
#   rowlayout:['quadruple', 'double', 'triple']
#   btn:['btn-orange', 'btn-inactive', 'btn-active']
#   text:['text','small-title',]
# }'

class DFSEnv():
    def __init__(self):
        self.reset()
        wordtype = {
            root:['root'],
            element:['header'],
            row:['row'],
            rowlayout:['quadruple', 'double', 'triple'],
            btn:['btn-orange','btn-green', 'btn-red', 'btn-inactive', 'btn-active'],
            text:['text', 'small-title'],
            end:['end']
        }
        self.rule = {
            root:['element', 'row', 'end'],
            element:['btn', 'text', 'end'],
            row:['rowlayout'],
            rowlayout:['element', 'end']
        }
    
    def state(self):
        if self.parent.value == 'root':
            
        return self.root, mask
        
    def step(self, action):
        return done, reward
    
    def reset(self):
        self.tree = Tree('root')
        self.parent = self.root
        
class BFSEnv():
    pass

class PathEnv():
    pass