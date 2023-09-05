from tinychad.tensor import tensor, Linear, Conv2d, BatchNorm2d
from tinychad.optim import SGD
from tinychad.helpers import get_parameters
import numpy as np

from mnist import mnist
from tqdm import tqdm, trange
import time


xtrain, ytrain, xtest, ytest = mnist('MNIST')

xtrain = xtrain / 255.0

class convchadnet: 
  def __init__(self): 
    self.c1 = Conv2d(1,32,5, stride=1, padding=2)
    self.c2 = Conv2d(32,64,5, stride=1, padding=2)
    self.l1 = Linear(7*7*64, 1000)
    self.l2 = Linear(1000, 10)

  def forward(self, x): 
    x = self.c1(x).relu()
    x = x.max_pool2d((2,2), stride=2)

    x = self.c2(x).relu()
    x = x.max_pool2d((2,2), stride=2)

    x = x.reshape(x.shape[0],-1)
    x = self.l1(x)
    x = self.l2(x)
    x = x.logsoftmax(axis=1)
    return x

  def get_parameters(self): 
    return get_parameters(self)

model = convchadnet() 
BS= 12
optim = SGD(model.get_parameters(), lr = 1e-4, momentum = 0.9)

def train(model, optim, xtrain, ytrain):
  for jj in (t := trange(1000)):
    ind = np.random.randint(0,59000, size=BS)
    inp = xtrain[ind][:].reshape(BS,1,28,28).astype(np.float32)
    out = model.forward(tensor(inp))
    res = ytrain[ind]
    loss = out.NLLLoss(res)

    cat = np.argmax(out.data, axis=1)
    acc = np.sum(cat == ytrain[ind])

    optim.zero_grad()
    loss.backward()
    optim.step()
    
    t.set_description("loss %.6f acc %.2f" % (loss.data, acc / BS))

train(model, optim, xtrain, ytrain)



