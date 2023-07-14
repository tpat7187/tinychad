import sys
sys.path.insert(1, '../')

from tinychad.tensor import tensor, Linear, Conv2d
from tinychad.optim import SGD
import numpy as np

from mnist import mnist
from tqdm import tqdm, trange
import time

xtrain, ytrain, xtest, ytest = mnist('MNIST')

class convchadnet: 
  def __init__(self): 
    self.c1 = Conv2d(1,3,3)
    self.c2 = Conv2d(3,1,5)

    self.l1 = Linear(100,50)
    self.l2 = Linear(50,10)

  def forward(self, x): 
    x = self.c1(x).avg_pool2d((3,3), stride=1).relu()
    x = self.c2(x).avg_pool2d((5,5), stride=2).relu()

    x = x.reshape(1,-1)

    x = self.l1(x).relu()
    x = self.l2(x)
    x = x.logsoftmax(axis=1)
    return x

model = convchadnet() 
BS= 1
params = [model.l1.w, model.l2.w, model.c1.w, model.c1.b, model.c2.w, model.c2.b]

optim = SGD(params, lr = 1e-3)

def train(model, optim, xtrain, ytrain):
  for jj in range(1000):
    ind = np.random.randint(0,59000, size=BS)
    inp = xtrain[0].reshape(1,1,28,28).astype(np.float32)
    out = model.forward(tensor(inp))
    res = ytrain[0]
    OHE = np.zeros((res.size, 10)) 
    OHE[np.arange(res.size), res] = 1
    res = tensor(OHE)


    loss = (-res.mul(out + 1e-10)).mean(axis=0).mean()

    cat = np.argmax(out.data, axis=1)
    acc = np.sum(cat == ytrain[0])

    #print(out.data)
    #print('pred', cat, 'actual', ytrain[0], loss.data)

    optim.zero_grad()
    loss.backward()
    optim.step()
    
    #t.set_description("loss %.6f acc %.2f" % (loss.data, acc / BS))

train(model, optim, xtrain, ytrain)



