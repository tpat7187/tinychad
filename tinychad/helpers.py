import numpy as np 
import inspect
from tinychad.tensor import tensor, Conv2d, Linear, BatchNorm2d
from typing import Union, Optional, Tuple

# blocks will be object -> several Linear/Conv2d/BatchNorm -> tensor, tensor
def get_parameters(obj:object) -> dict: 
    layers, params = (Conv2d, Linear, BatchNorm2d), []
    states = inspect.getmembers(obj, lambda a: not(inspect.isroutine(a))), []
    states = [a for a in states[0] if not(a[0].startswith('__') and a[0].endswith('__'))]

    def check_for_tensor(obj):
        states = inspect.getmembers(obj, lambda a: not(inspect.isroutine(a)))
        states = [a for a in states if not(a[0].startswith('__') and a[0].endswith('__'))]
        [params.append(j[1]) for _,j in enumerate(states) if type(j[1]) == tensor]

    def check_for_layer(obj):
        states = inspect.getmembers(obj, lambda a: not(inspect.isroutine(a))), []
        states = [a for a in states[0] if not(a[0].startswith('__') and a[0].endswith('__'))]
        out = [j for _,j in enumerate(states) if type(j[1]) in layers]
        [check_for_tensor(j[1]) if len(out) > 0 else [] for j in states]

    for s in states: 
        if isinstance(s[1], tensor): 
            params.append(s[1])
        elif isinstance(s[1], (layers)): 
            check_for_tensor(s[1])
        else: 
            check_for_layer(s[1])

    return params

