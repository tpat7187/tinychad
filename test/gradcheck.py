import numpy as np
import sys

sys.path.insert(1, '../')


from tinychad.tensor import tensor
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
  l2 = (l1 / z - b2).sum()
  l3 = l2.log().exp().reshape(-1,1).logsoftmax(axis=1).sum()
 
  print(l2.data)
  l3.backward()

  print(x.grad)
  print(y.grad)
  print(z.grad)

  grads = np.array([x.grad, y.grad, z.grad])
 
  return grads

def torchTEST():
  x = torch.tensor(XIN, requires_grad = True)
  y = torch.tensor(YIN, requires_grad = True)
  z = torch.tensor(ZIN, requires_grad = True)
  b = torch.tensor(BIN, requires_grad = True)
  b2 = torch.tensor(BIN, requires_grad = True)

  l1 = x @ y + b
  l2 = (l1 / z - b2).sum()
  l3 = l2.log().exp().reshape(-1,1).log_softmax(axis=1).sum()


  print(l2)

  l3.backward()

  print(x.grad)
  print(y.grad)
  print(z.grad)


  grads = np.array([x.grad.numpy(), y.grad.numpy(), z.grad.numpy()])

  return grads

def LSM(): 
  from tinygrad.tensor import Tensor


  N = np.random.randn(5,5).astype(np.float32)

  xt = torch.tensor(N, requires_grad = True)
  x = tensor(N)
  xtg = Tensor(N) 

  y = x.softmax(axis=1)
  yt = xt.softmax(axis=1)
  ytg = xtg.softmax(axis=1)

  print(y.data)
  print(yt)
  print(ytg.numpy())


def MNIST(): 
  W1 = np.random.randn(28*28, 128).astype(np.float32)
  W2 = np.random.randn(128, 10).astype(np.float32)
  B1 = np.random.randn(128).astype(np.float32)
  B2 = np.random.randn(10).astype(np.float32)

  inp = tensor.randn(28,28).reshape(1,-1)
  w1 = tensor.randn(28*28, 128)
  b1 = tensor.randn(128)
  w2 = tensor.randn(128,10)
  b2 = tensor.randn(10)


  a = (inp.dot(w1) + b1).relu()
  b = (a.dot(w2) + b2).relu()
  c = b.logsoftmax(axis=1)
  d = c.sum()

  d.backward()





if __name__ == "__main__":
  MNIST()









  



 



