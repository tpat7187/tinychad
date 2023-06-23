from tinychad.tensor import tensor
import numpy as np 

class Optimizer(): 
  def __init__(self, params, lr=1e-3):
    self.params = params
    #self.buffers = buffers

  def zero_grad(self): 
    for param in self.params: 
      param.grad = np.zeros(param.grad.shape, dtype = np.float32)

class SGD(Optimizer): 
  def __init__(self, params, lr=1e-3):
    super().__init__(params, lr)
    self.lr = lr

  def step(self): 
    for param in self.params: 
      param.data = param.data - self.lr*param.grad

# is tinygrad delting the grads if they're not needed?




