from __future__ import annotations
from typing import Union, List
from tinychad.tensor import tensor
import numpy as np 

class Optimizer(): 
  def __init__(self, params, lr=1e-3):
    self.params = params
    for param in params: 
      param.requires_grad = True

  def zero_grad(self): 
    for param in self.params: 
      #param.grad = np.zeros(param.shape, dtype = np.float32)
      param.grad = None


class SGD(Optimizer): 
  def __init__(self, params:list[tensor], momentum=0, lr=1e-3):
    super().__init__(params, lr)
    self.lr, self.momentum = lr, momentum
    self.v = [np.zeros(i.shape) for i in self.params] if self.momentum else []

  def step(self): 
    for i, param in enumerate(self.params):
      if self.momentum:
        self.v[i] = (self.momentum * self.v[i]) + param.grad
        param.grad = self.v[i]
      param.data = param.data - self.lr*param.grad

