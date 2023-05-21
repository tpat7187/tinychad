from tinychad.tensor import tensor
import torch

inp = tensor.randn(28,28,10)
inp = inp.reshape(10,-1)

l1_w = tensor.randn(784,128) 
l1_b = tensor.randn(10,128)

l2_w = tensor.randn(128,10)
l2_b = tensor.randn(10,10)

layer1 = (inp @ l1_w + l1_b).relu()
layer2 = (layer1 @ l2_w + l2_b).relu()
layer3 = layer2.logsoftmax(axis=1).sum(axis=1).sum()

layer3.backward()






















