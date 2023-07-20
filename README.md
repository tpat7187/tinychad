tinychad be a deep learning framework, not a good deep learning framework

it is gonna be slow, but its gonna be based

goals: 
  - write a compiler that compiles tinychad code into WASM OR into something more based
  - recreate important deep learning architectures using tinychad (LSTM, RESNET, Transformer)

TODO: 
  * computation graph
  * look into LLVMlite (how can we recreate matmul and other binary ops in LLVM IR, WHAT ABOUT MLIR) 
    * lazy evaluation-esque runtime for assigning buffers and shapes
  * convolutions
    * fix pooling (use reshape method over im2col [fix MAX backward for axis > 1]
    * Write efficientNet (HOT DOG, RED HOT) 
  * state dict for transfer learning

things to read: 
  - https://llvmlite.readthedocs.io/en/latest/ "llvmlite documentation" 
  - https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster "im2col"


