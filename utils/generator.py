import json
#import random
from .tree import Tree

class Env():
    def __init__(self, rule='rule.json'):
        with open(rule) as data:
            self.rule = json.load(data)

        self.stack = []
        self.start_token = 'root'
        self.end_token = 'end'
        
        # build type list
        self.type_list = dict()
        for tag, tag_type in self.rule['dict'].items():
            if tag_type in self.type_list:
                self.type_list[tag_type].append(tag)
            else:
                self.type_list[tag_type] = [tag]
                
        self.reset()


    def reset(self):
        self.root = Tree(self.start_token)
        self.stack = [self.root]

    def step(self, action):
        new_node = Tree(action)
        self.stack[-1].add_child(new_node)
        pointer_type = self.rule['dict'][self.stack[-1].value]
        child_limit = self.rule['type'][pointer_type]['child'][0]['limit']
        
        if action==self.end_token:
            self.stack.pop()
            if len(self.stack) == 0:
                return self.root, None, []
            else:
                return self.state()
            
        if len(self.stack[-1].children) >= child_limit-1:
            self.stack[-1].add_child(Tree(self.end_token))
            self.stack.pop()

#        action_type = self.rule['dict'][action]
#        if self.rule['type'][action_type]['child'][0]['limit'] > 0:
        self.stack.append(new_node)
                
        if len(self.stack) == 0:
            return self.root, None, []
        else:
            return self.state()


    def state(self):
        # return root, point to the node building child, chioce you can choose
        chioce = [self.end_token]
        pointer_type = self.rule['dict'][self.stack[-1].value]
        for tag_type in self.rule['type'][pointer_type]['child'][0]['type']:
            chioce += self.type_list[tag_type]
        return self.root, self.stack[-1], chioce

#env = Env()
#
#env.reset()
#tree, pointer, choice = env.state()
#while pointer != None:
#    tree, pointer, choice = env.step(random.choice(choice))
#    
#print(tree)
##%% write to file and make to jpg
#stack = [tree]
#out_data = []
#
#while len(stack) != 0:
##    print(len(stack))
#    if stack[-1].value == 'end':
#        stack.pop()
#        out_data.append( '/' + stack.pop().value)
#    else:
#        out_data.append(stack[-1].value)
#        stack += list(reversed(stack[-1].children))
#
#out_data = ['<{}>'.format(i) for i in out_data]
#with open('out.xml','w') as f:
#    f.write(''.join(out_data))
#
#import os
#os.system('node compiler/screenshot.js out.xml out.jpg compiler/template.html')