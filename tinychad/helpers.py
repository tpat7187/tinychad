import numpy as np 
import inspect
from tinychad.tensor import tensor
from typing import Union, Optional, Tuple


# TODO: check object, if object contains no tensors loop through object until it finds layers OR tensors 
# if layers -> look for ['w'], ['b'] in object dict
# if tensors -> look for ['requries_grad'] in object dict

# blocks will be object -> several Linear/Conv2d/BatchNorm -> tensor, tensor
def get_parameters(obj:object) -> dict: 
    states, params = inspect.getmembers(obj, lambda a: not(inspect.isroutine(a))), []
    states = [a for a in states if not(a[0].startswith('__') and a[0].endswith('__'))]
    for s in states:
        if type(s[1]) == tensor: params.append(s[1])
        else: 
            states = inspect.getmembers(s[1], lambda a: not(inspect.isroutine(a)))
            states = dict([a for a in states if not(a[0].startswith('__') and a[0].endswith('__'))])
            params.append(states['w'])
            params.append(states['b']) if states['b'] is not None else states['b']
    return params