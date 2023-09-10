tinychad be a deep learning framework, not a good deep learning framework

it is gonna be slow, but its gonna be based

goals: 
  - write a compiler that compiles tinychad code into WASM OR into something more based
  - recreate important deep learning architectures using tinychad (LSTM, RESNET, Transformer)

TODO: 
  * computation graph
  * LLVM/MLIR backend
    * may still be value in tinychad -> MLIR -> LLVM IR -> WASM
    * MNIST IN THE WEB
  * update training/testing
    * batchnorm2d passing tests
  * state dict for transfer learning

## how to chad
tinychad is like pytorch but slower but also significantly smaller
```python
from tinychad.tensor import tensor, Linear
from tinychad.optim import SGD

inp = tensor.randn(3,28,28)
class CHADnet:
  def __init__(self):
    self.l1 = Linear(784, 128)
    self.l2 = Linear(128, 10, bias=False)
  def forward(self, x):
    x = x.reshape(-1,784)
    x = self.l1(x).relu()
    x = self.l2(x).relu()
    x = x.logsoftmax(axis=1)
    return x

model = CHADnet()
model.forward(inp)
```

tinychad is now lazy (execute with LAZY=1) 

```python
from tinychad.tensor import tensor

x = tensor.randn(1024,1024)
y = tensor.randn(1024,1024)

w = x.matmul(y) # operations get automatically stored in cache

w.exec() # to execute the cache
w.get_buffers() # to return the list of unique buffers and shapes needed for the output to be realized
```



things to read: 
  - https://llvmlite.readthedocs.io/en/latest/ "llvmlite documentation" 
  - https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster "im2col"


