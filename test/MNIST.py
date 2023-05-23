import numpy as np
import torch.nn as nn
import torch
from tinychad.tensor import tensor
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
    x = tensor(x)
    x = (x.dot(self.w1))
    x = (x.dot(self.w2))
    x = x.logsoftmax(axis=0)


    return x

class torchnet(nn.Module): 
  def __init__(self): 
    super().__init__()
    self.w1 = torch.tensor(W1)
    self.b1 = torch.tensor(B1)

    self.w2 = torch.tensor(W2)
    self.b2 = torch.tensor(B2)


    self.LSM = nn.LogSoftmax(dim=0)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = torch.tensor(x)
    x = x @ self.w1 + self.b1
    x = self.relu(x)
    x = x @ self.w2 + self.b2
    x = self.LSM(x)
    return x



# torch handles NANs and infinites differently to the way we do 
def train(xtrain, ytrain, model):
  BS = 10
  for ff in (t := trange(1)):
    ind = np.random.randint(0,59000, size=BS) 
    inp = xtrain[np.array([ind])[:]].reshape(BS,-1)
    out = model.forward(inp).sum()


    out.backward()


model = bobnet()
train(xtrain, ytrain, model)



