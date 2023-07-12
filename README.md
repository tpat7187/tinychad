tinychad be a deep learning framework, not a good deep learning framework

it is gonna be slow, but its gonna be based

goals: 
  - write a compiler that compiles tinychad code into WASM OR into something more based
  - recreate important deep learning architectures using tinychad (LSTM, RESNET, Transformer)

TODO: 
  * computation graph
  * look into LLVMlite (how can we recreate matmul and other binary ops in LLVM IR)
    * lazy evaluation-esque runtime for assigning buffers and shapes
  * convolutions
    * write tests
    * convNet for MNIST? (need to write pooling + other non-linearity for image processing)
    * Write efficientNet (HOT DOG, RED HOT)
  * re-write DEBUG to show in/out shapes AND time it takes to execut the kernel

things to read: 
  - https://llvmlite.readthedocs.io/en/latest/ "llvmlite documentation" 



