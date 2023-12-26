import numpy as np 
import inspect, os, shlex
from typing import Union, Optional, Tuple
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS, LoadOPS
import networkx as nx
import matplotlib.pyplot as plt

# blocks will be object -> several Linear/Conv2d/BatchNorm -> tensor, tensor
# increase depth of tensor search 
def get_parameters(obj:object, max_depth:int=4, depth:int=0) -> list: 
    from tinychad.tensor import tensor, Conv2d, Linear, BatchNorm2d

    if depth > max_depth:
        return []
    layers, params = (Conv2d, Linear, BatchNorm2d), []
    states = inspect.getmembers(obj, lambda a: not(inspect.isroutine(a)))
    states = [a for a in states if not(a[0].startswith('__') and a[0].endswith('__'))]

    for s in states: 
        if isinstance(s[1], tensor):
            params.append(s[1])
        elif isinstance(s[1], layers):
            params.extend(get_parameters(s[1], max_depth, depth + 1))
        elif hasattr(s[1], '__dict__'): 
            params.extend(get_parameters(s[1], max_depth, depth + 1))
    return params

def generate_graph(runner):
    G = nx.DiGraph()
    def _populate_graph(node, G):
        label = f"\"{node.bufferList[-1].shape}\n\"" if node.opList[-1] not in LoadOPS else f"\"{node.bufferList[-1].shape}\n {node.bufferList[-1].op}\""
        G.add_node(node.id, label=label)
        for child in node.children:
            edge_label = " ".join(shlex.quote(str(op)) for op in child.opList if op and op not in LoadOPS)
            G.add_edge(child.id, node.id, label=edge_label)
            _populate_graph(child, G)
    _populate_graph(runner.root, G)
    dot_path = f'{runner.graph_path}.dot'
    nx.drawing.nx_pydot.write_dot(G, dot_path)
    svg_path = f"{runner.graph_path}.svg"
    cmd = f'dot -Tsvg -Grankdir=BT "{dot_path}" -o "{svg_path}"'
    return_code = os.system(cmd)
    if return_code != 0: print(f"An error occurred while generating the graph. Return code: {return_code}")
    else: print(f"Graph saved to {svg_path}")