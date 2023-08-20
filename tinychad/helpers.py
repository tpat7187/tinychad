import numpy as np 
import inspect
from tinychad.tensor import tensor, Conv2d, Linear, BatchNorm2d
from typing import Union, Optional, Tuple

# blocks will be object -> several Linear/Conv2d/BatchNorm -> tensor, tensor
def get_parameters(obj:object) -> dict: 
    layers, params = (Conv2d, Linear, BatchNorm2d), []
    def check_for_tensor(obj):
        states = inspect.getmembers(obj, lambda a: not(inspect.isroutine(a)))
        states = [a for a in states if not(a[0].startswith('__') and a[0].endswith('__'))]
        [params.append(j[1]) for _,j in enumerate(states) if type(j[1]) == tensor]

    def check_for_layer(obj):
        states = inspect.getmembers(obj, lambda a: not(inspect.isroutine(a))), []
        states = [a for a in states[0] if not(a[0].startswith('__') and a[0].endswith('__'))]
        out = [j for _,j in enumerate(states) if type(j[1]) in layers]
        # if no layers could just contain tensors
        [check_for_layer(j[1]) if len(out) == 0 else check_for_tensor(j[1]) for j in states]


    check_for_layer(obj) # grab layers and iterate through composite layers
    check_for_tensor(obj) # grab tensors
    return params