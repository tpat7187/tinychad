from tinychad.tensor import tensor
import tinychad.ops as ops
import torch


''' 
the idea: multibatching during MNIST 

bias: 1x128 
weight@input: 10x128 
biasis gonna get broadcasted to 10x128 
during backrpop, grad is gonna be 10x128
cannot fit 10x128 grad into 1x128

algo should check shapes and if they are unbroadcastable, then they should be unbroadcasted


(5,1) : (5,5) -> (5,1) (5,1) 

(5,5) : (5,5) -> (5,5) (5,5)

'''

out = tensor.randn(5,5).data
saved = tensor.randn(5,1).data
print(f"out = {out.shape}, saved ={saved.shape}")

out = ops._unbr(out,saved)

print(f"out = {out.shape}, saved ={saved.shape}")

# take in input tensor and output shape
# if broadcastbale 
# unbroadcast to output shape


























