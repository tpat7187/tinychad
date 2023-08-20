import numpy as np 
import inspect
from tinychad.tensor import tensor, Conv2d, Linear, BatchNorm2d
from typing import Union, Optional, Tuple

# blocks will be object -> several Linear/Conv2d/BatchNorm -> tensor, tensor
# increase depth of tensor search 
def get_parameters(obj:object, max_depth:int=4, depth:int=0) -> list: 
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