import numpy as np 
import inspect, os, shlex
from typing import Union, Optional, Tuple
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS, LoadOPS
import networkx as nx

# blocks will be object -> several Linear/Conv2d/BatchNorm -> tensor, tensor
# increase depth of tensor search 

# TODO: make this its own class
# measure time in frontend, backend and execution
DEBUG = os.getenv("DEBUG") 

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
    graph_path = '/tmp/ast_graph'
    G = nx.DiGraph()

    def _populate_graph(node, G, visited):
        if id(node) in visited:
            return
        visited.add(id(node))
        
        label = f"\{node.shape}\n"
        #label += ', '.join([str(j.name) for j in node.op]) if isinstance(node.op, list) else str(node.op.name)
        label += ', '.join([str(j.name) for j in node.kernArgs])
        G.add_node(id(node), label=label)


        if node.children is not None:
            for child in node.children:
                G.add_edge(id(child), id(node))
                _populate_graph(child, G, visited)
    visited = set()
    _populate_graph(runner, G, visited)
    dot_path = f'{graph_path}.dot'
    nx.drawing.nx_pydot.write_dot(G, dot_path)
    svg_path = f"{graph_path}.svg"
    cmd = f'dot -Tsvg -Grankdir=BT "{dot_path}" -o "{svg_path}"'
    return_code = os.system(cmd)
    if return_code != 0:
        print(f"An error occurred while generating the graph. Return code: {return_code}")
    else:
        print(f"Graph saved to {svg_path}")
