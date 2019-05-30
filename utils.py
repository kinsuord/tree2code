# -*- coding: utf-8 -*-
from tree import Tree
import numpy as np
import os

''' change json_dict to tree structure.'''
def json2tree(json_dict):
    root = Tree('root')
    _json2tree_rec(json_dict, root)
    return root

''' recursively change json_dict to tree. '''
def _json2tree_rec(json_dict, tree):
    if tree.value=='root':
        for child_json in json_dict['children']:
            child_tree = Tree(child_json['type'])
            tree.add_child(child_tree)
            _json2tree_rec(child_json, child_tree)
    
    elif tree.value=='element':
        tree.add_child(Tree(json_dict['tagName']))
        for key, value in json_dict['properties'].items():
            key_node = Tree(key)
            tree.add_child(key_node)
            _json2tree_rec(value, key_node)
        tree.add_child(Tree('None'))
        for child_json in json_dict['children']:
            child_tree = Tree(child_json['type'])
            tree.add_child(child_tree)
            _json2tree_rec(child_json, child_tree)        
            
    elif tree.value=='text':
        tree.add_child(Tree(json_dict['value']))
    
    # property key
    else:
        if isinstance(json_dict, list):
            tree.add_child(Tree(' '.join(json_dict)))
        else:
            tree.add_child(Tree(str(json_dict)))

def vec2word(tree, word_dict):
    
    for key in word_dict.keys():
        if np.array_equal(tree.value, word_dict[key]):
            tree.value = key

    for child in tree.children:
        vec2word(child, word_dict)
    return tree   

def dsl2tree(dsl):
    root = Tree('root')
    words = dsl.split()
    stack = [root]
    last_child = None
    for word in words:
        if word == '{':
            stack.append(last_child)
        elif word == '}':
            child = Tree('None')
            stack[-1].add_child(child)
            stack.pop()
        else:
            child = Tree(word.replace(',', ''))
            stack[-1].add_child(child)
            last_child = child
    stack[-1].add_child(Tree('None'))
    return root
       
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)
#with open('dataset/pix2code/dsl/10.gui') as f:
#    dsl_code = f.read()
#tree = dsl2tree(dsl_code)