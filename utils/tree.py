import networkx as nx
import matplotlib.pyplot as plt

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
    
    def copy(self):
        ''' return the new one with the same value'''
        copy_tree = Tree(self.value)
        for child in self.children:
            copy_tree.add_child(child.copy())
        return copy_tree
    
    def for_each_value(self, func):
        ''' do something for each node in tree. 
            EX:tree.for_each_value(lambda x: x.to(torch.device('cpu')))
        '''
        self.value = func(self.value)
        for child in self.children:
            child.for_each_value(func)
        
    def __str__(self):
        '''print the tree'''
        out_str = '  '*self.height() + self.value.__str__()
        for child in self.children:
            out_str += '\n' + child.__str__()
        return out_str

def tree_similarity(tree1, tree2):
    def tree_to_graph(tree):
        node_id = 1
        queue = []
        graph = nx.Graph()
    
        queue.append({'element': tree, 'id': node_id})
        graph.add_node(node_id, element=tree.value)
        node_id += 1
        
        while len(queue):
            parent = queue[0]
            for child in parent['element'].children:
                graph.add_node(node_id, element=child.value)
                graph.add_edge(parent['id'], node_id)
                queue.append({'element': child, 'id': node_id})
                node_id += 1
            queue.pop(0)
            
        return graph
    
    graph1 = tree_to_graph(tree1)
    graph2 = tree_to_graph(tree2)
    # plt.title("graph1")
    # nx.draw_networkx(graph1)
    # plt.show()
    # plt.title("graph2")
    # nx.draw_networkx(graph2)
    # plt.show() 
    def get_element_tuple(graph, path):
        arr=[]
        for i in path:
            arr.append(graph.nodes[i]['element'])
        return tuple(arr)

    def generate_subpaths(path, l, graph, subpath_track):
        if l >= len(path):
            tuple_path=get_element_tuple(graph, path)
            # tuple_path=tuple(path)
            if tuple_path not in subpath_track:
                subpath_track[tuple_path] = 1
            else:
                subpath_track[tuple_path] += 1
        else:
            index = 0
            while l+index-1 < len(path):
                tuple_path=get_element_tuple(graph, path[index: l+index])
                # tuple_path=tuple(path[index: l+index])
                if tuple_path not in subpath_track:
                    subpath_track[tuple_path] = 1
                else:
                    subpath_track[tuple_path] += 1
                index += 1
    
            generate_subpaths(path, l+1, graph, subpath_track)
    def get_subpaths(graph, root, track, path, subpath_track):
        track[root] = True # record visited nodes
        if graph.degree(root) == 1:
            
            generate_subpaths(path, 1, graph, subpath_track)
        else:
            for node in graph.neighbors(root):
                if node not in track: # if node not visited
                    get_subpaths(graph, node, track, path + [node, ], subpath_track)
                
    def get_kernel(subpath_track_1, subpath_track_2):
        decay_rate=0.75
        kernel_v=0
        for p in subpath_track_1:
            for q in subpath_track_2:
                if p==q:    
                    kernel_v+=subpath_track_1[p]*subpath_track_2[q]/pow(decay_rate, len(q)-1)
        return kernel_v

    def get_normalized_kernel(subpath_track_1, subpath_track_2):
        kernel_12 = get_kernel(subpath_track_1, subpath_track_2)
        kernel_1 = get_kernel(subpath_track_1, subpath_track_1)
        kernel_2 = get_kernel(subpath_track_2, subpath_track_2)
        if kernel_1 < kernel_2:
            kernel_1=kernel_2
        return kernel_12/kernel_1
   
    subpath_track = {}
    track={}
    path=[]
    get_subpaths(graph1, 1, track ,path, subpath_track)
    subpath_track_1=subpath_track

    subpath_track = {}
    track={}
    path=[]
    get_subpaths(graph2, 1, track ,path, subpath_track)
    subpath_track_2=subpath_track

    kernel_v = get_normalized_kernel(subpath_track_1, subpath_track_2)
    return kernel_v