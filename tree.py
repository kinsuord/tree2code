# -*- coding: utf-8 -*-

class Tree(object):
    def __init__(self, value):
        self.parent = None
        self.value = value
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if hasattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if hasattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth
    
    def height(self):
        if hasattr(self, '_height'):
            return self._height
        if self.parent == None:
            self._height = 0
        else:
            self._height = self.parent.height() + 1
        
        return self._height
        
    def __str__(self):
        out_str = '  '*self.height() + self.value.__str__()
        for child in self.children:
            out_str += '\n' + child.__str__()
        return out_str
    