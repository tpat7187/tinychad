import sys
sys.path.insert(1, '../')

import numpy as np
import torch.nn as nn
import torch
from tinychad.tensor import tensor, Linear
from tinychad.optim import SGD
from mnist import mnist
from tqdm import tqdm, trange
import time

xtrain, ytrain, xtest, ytest = mnist('MNIST')

class bobnet(): 
  def __init__(self): 
    self.l1 = Linear(28*28, 128)
    self.l2 = Linear(128, 10)

  def forward(self, x): 

    x = self.l1(x)
    x = self.l2(x)
    x = x.logsoftmax(axis=1)
    return x

BS= 32
model = bobnet()
optim = SGD([model.w1, model.w2, model.b1, model.b2], lr = 1e-3)

def train(model, optim, xtrain, ytrain):
  for jj in (t := trange(10000)):
    ind = np.random.randint(0,59000, size=BS)
    inp = xtrain[ind][:].reshape(BS,-1).astype(np.float32)
    out = model.forward(tensor(inp))
    res = ytrain[ind]
    OHE = np.zeros((res.size, 10)) 
    OHE[np.arange(res.size), res] = 1
    res = tensor(OHE)

    loss = (-res.mul(out + 1e-10)).mean(axis=0).mean()

    cat = np.argmax(out.data, axis=1)
    acc = np.sum(cat == ytrain[ind])

    optim.zero_grad()
    loss.backward()
    optim.step()

    t.set_description("loss %.6f acc %.2f" % (loss.data, acc / BS))

train(model, optim, xtrain, ytrain)

















