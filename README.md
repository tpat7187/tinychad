tinychad be a deep learning framework, not a good deep learning framework

it is gonna be slow, but its gonna be based

goals: 
  - write a compiler that compiles tinychad code into WASM OR into something more based
  - recreate important deep learning architectures using tinychad (LSTM, RESNET, Transformer)

TODO: 
  * computation graph
  * LLVM/MLIR backend
    * test llvmlite/ctypes with lazybuffers
    * MLIR/LLVM kelido toy-dialect
  * update training/testing
    * match torch on NLLL and CrossEntropyLoss
    * convnet for MNIST
  * state dict for transfer learning

things to read: 
  - https://llvmlite.readthedocs.io/en/latest/ "llvmlite documentation" 
  - https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster "im2col"


