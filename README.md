tinychad be a deep learning framework, not a good deep learning framework

it is gonna be slow, but its gonna be based

goals: 
  - write a compiler that compiles tinychad code into WASM OR into something more based
  - recreate important deep learning architectures using tinychad (LSTM, RESNET, Transformer)

TODO: 
  * elementwise op kernel fusion
  * have backend reuse kernels
  * write matmul kernel as a (reshape * transpose.reshape).sum()
  * Compiled MNIST
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
    return x.realize()

model = CHADnet()
model.forward(inp)
```



