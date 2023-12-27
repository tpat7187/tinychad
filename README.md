tinychad be a deep learning framework, not a good deep learning framework

it is gonna be slow, but its gonna be based

goals: 
  - write a compiler that compiles tinychad code into WASM OR into something more based
  - recreate important deep learning architectures using tinychad (LSTM, RESNET, Transformer)

TODO: 
  * rewrite frontend and allow for simple graph opts on the computation graph
    * elementwise kernel fusion
    * reshapes dont make copies
  * write tokenizier for computation graph nodes
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

tinychad can generate LLVM IR (execute with LLVM=1) 

```python
from tinychad.tensor import tensor, Linear

x = tensor.randn(5,5).reshape(1, -1)
l1 = Linear(25, 10)

out = l1(x).exec() # execute cache with LLVM codegen
```

to export the model run PORT=1 (this will load all the buffers into the LLVM module instead of reading the pointers from the Buffers)

```LLVM
; ModuleID = ""
target triple = "unknown-unknown-unknown"
target datalayout = ""

define void @"main"(float* %".1", float* %".2", float* %".3", float* %".4", float* %".5", float* %".6", float* %".7")
{
buffers:
  br label %"main"
main:
  call void @"ReshapeOPS.RESHAPE_5_5_1_25"(float* %".1", float* %".4")
  call void @"BinaryOPS.MATMUL_1_25_25_10"(float* %".4", float* %".2", float* %".5")
  call void @"ReshapeOPS.RESHAPE_10_1_10"(float* %".3", float* %".6")
  call void @"BinaryOPS.ADD_1_10"(float* %".5", float* %".6", float* %".7")
  br label %"exit"
exit:
  ret void
}
```


things to read: 
  - https://llvmlite.readthedocs.io/en/latest/ "llvmlite documentation" 
  - https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster "im2col"


