import numpy as np
import torch.nn as nn
import torch
from tinychad.tensor import tensor, Linear
from tinychad.optim import SGD
from tinychad.helpers import get_parameters
from mnist import mnist
from tqdm import tqdm, trange
import time

xtrain, ytrain, xtest, ytest = mnist('MNIST')
xtrain = xtrain / 255.0


class bobnet(): 
  def __init__(self): 
    self.l1 = Linear(28*28, 128)
    self.l2 = Linear(128, 10)

  def forward(self, x): 

    x = self.l1(x)
    x = self.l2(x)
    x = x.logsoftmax(axis=1)
    return x

  def get_parameters(self): 
    return get_parameters(self)

BS= 128
model = bobnet()
optim = SGD(model.get_parameters(), lr = 1e-3, momentum=0.9)

def train(model, optim, xtrain, ytrain):
  for jj in (t := trange(3000)):
    ind = np.random.randint(0,59000, size=BS)
    inp = xtrain[ind][:].reshape(BS,-1).astype(np.float32)
    out = model.forward(tensor(inp))
    res = tensor(ytrain[ind])
    loss = out.NLLLoss(res)
    cat = np.argmax(out.detach(), axis=1)
    acc = np.sum(cat == ytrain[ind])

    optim.zero_grad()
    loss.backward()
    optim.step()

    t.set_description("loss %.6f acc %.2f" % (loss.detach(), acc / BS))

train(model, optim, xtrain, ytrain)

















