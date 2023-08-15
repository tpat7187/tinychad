import sys
sys.path.insert(1, '../')

from tinychad.tensor import tensor, Linear, Conv2d, BatchNorm2d
from tinychad.optim import SGD
import numpy as np

from mnist import mnist
from tqdm import tqdm, trange
import time

xtrain, ytrain, xtest, ytest = mnist('MNIST')

class convchadnet: 
  def __init__(self): 
    self.c1 = Conv2d(1,16,5, stride=1, padding=2)
    self.b1 = BatchNorm2d(16)
    self.c2 = Conv2d(16,32,5, stride=1, padding=2)
    self.b2 = BatchNorm2d(32)
    self.l1 = Linear(7*7*32, 10)


  def forward(self, x): 
    x = self.c1(x)
    x = self.b1(x).relu()
    x = x.max_pool2d((2,2), stride=2)

    x = self.c2(x)
    x = self.b2(x).relu()
    x = x.max_pool2d((2,2), stride=2)

    x = x.reshape(x.shape[0],-1)
    x = self.l1(x)

    x = x.logsoftmax()

    return x

model = convchadnet() 
BS= 64
params = [model.l1.w, model.l1.b, model.c1.w, model.c1.b, model.c2.w, model.c2.b, model.b1.w, model.b2.w, model.b1.b, model.b2.b]
optim = SGD(params, lr = 1e-3)

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



