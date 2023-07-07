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
    * should conv be its own operation

things to read: 
  - https://llvmlite.readthedocs.io/en/latest/ "llvmlite documentation" 



