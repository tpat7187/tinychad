import numpy as np
from tensor import tensor
import torch

XIN = np.ones((3,3), dtype = np.float32)
YIN = np.ones((3,3), dtype = np.float32)
ZIN = np.ones((3,3), dtype = np.float32)
BIN = np.ones((3,3), dtype = np.float32)

def tinychadTEST():
  x = tensor(XIN)
  y = tensor(YIN)
  z = tensor(ZIN)
  b = tensor(BIN)
  b2 = tensor(BIN)

  l1 = x @ y + b
  l2 = l1 / z - b2
  l3 = l2.log().exp().relu().reshape(-1,1).sum(axis=1).sum()

  l3.backward()

  grads = np.array([x.grad, y.grad, z.grad])
 
  return grads

def torchTEST():
  x = torch.tensor(XIN, requires_grad = True)
  y = torch.tensor(YIN, requires_grad = True)
  z = torch.tensor(ZIN, requires_grad = True)
  b = torch.tensor(BIN, requires_grad = True)
  b2 = torch.tensor(BIN, requires_grad = True)

  l1 = x @ y + b
  l2 = l1 / z - b2
  l3 = l2.log().exp().relu().reshape(-1,1).sum()

  l3.backward()
  grads = np.array([x.grad.numpy(), y.grad.numpy(), z.grad.numpy()])

  return grads

def LSM(): 
  x = tensor.randn(5,5)
  y = x.logsoftmax().sum()

  y.toposort(track = True)

  y.backward()




if __name__ == "__main__":
  to = torchTEST() 
  ti = tinychadTEST()
  
  print((to == ti).all())



  



 



