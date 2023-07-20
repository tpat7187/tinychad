from tinychad.tensor import tensor
import numpy as np 

class Optimizer(): 
  def __init__(self, params, lr=1e-3):
    self.params = params
    for param in params: 
      param.requires_grad = True

  def zero_grad(self): 
    for param in self.params: 
      param.grad = np.zeros(param.grad.shape, dtype = np.float32)

class SGD(Optimizer): 
  def __init__(self, params, lr=1e-3, momentum=0):
    super().__init__(params, lr)
    self.lr = lr

  def step(self): 
    for param in self.params: 
      param.data = param.data - param.grad*self.lr





