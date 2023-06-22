
import sys
sys.path.insert(1, '../')
import numpy as np
import torch.nn as nn
import torch
from tinychad.tensor import tensor
from tinychad.optim import SGD

from mnist import mnist
from tqdm import tqdm, trange
import time

xtrain, ytrain, xtest, ytest = mnist('MNIST')

W1 = np.random.randn(28*28, 128).astype(np.float32)
W2 = np.random.randn(128, 10).astype(np.float32)
B1 = np.random.randn(128).astype(np.float32)
B2 = np.random.randn(10).astype(np.float32)

class bobnet(): 
  def __init__(self): 
    self.w1 = tensor(W1)
    self.b1 = tensor(B1)

    self.w2 = tensor(W2)
    self.b2 = tensor(B2)


  def forward(self, x): 
    x = tensor(x).reshape(1,-1)

    x = (x.dot(self.w1)).relu()
    x = (x.dot(self.w2)).relu()

    x = x.logsoftmax(axis=1)

    return x


model = bobnet()
inp = np.random.randn(28,28)
optim = SGD([model.w1, model.w2], lr = 1e-3)

#def train(model, optim, xtrain, ytrain, lr):
for jj in range(100):
  out = model.forward(inp)

  res = tensor([1,0,0,0,0,0,0,0,0,0])

  loss = (-res * (out)).mean(axis=1).mean()

  optim.zero_grad()
  loss.backward()
  optim.step()

  print(loss.data[0])





