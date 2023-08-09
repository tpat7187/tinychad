import sys

sys.path.insert(1, '../')

from tinychad.tensor import tensor, Linear
from tinychad.optim import SGD
from extra.training import sparse_categorical_crossentropy
import numpy as np
import torch
import torch.nn as nn 
import unittest


L1_w = np.random.randn(3,5).astype(np.float32)
L1_b = np.random.randn(5).astype(np.float32)

L2_w = np.random.randn(5,3).astype(np.float32)
L2_b = np.random.randn(3).astype(np.float32)


class bobnet: 
  def __init__(self): 
    self.l1w = tensor(L1_w)
    self.l1b = tensor(L1_b)
    self.l2w = tensor(L2_w)
    self.l2b = tensor(L2_b)

  def forward(self, x):
    x = x.dot(self.l1w).add(self.l1b).relu() 
    x = x.dot(self.l2w).add(self.l2b).relu() 
    x = x.logsoftmax(axis=1)
    return x

class torchnet: 
  def __init__(self):
    self.l1w = torch.tensor(L1_w, requires_grad = True)
    self.l1b = torch.tensor(L1_b, requires_grad = True)
    self.l2w = torch.tensor(L2_w, requires_grad = True)
    self.l2b = torch.tensor(L2_b, requires_grad = True)

  def forward(self, x):
    x = x.matmul(self.l1w).add(self.l1b).relu()
    x = x.matmul(self.l2w).add(self.l2b).relu()
    x = x.log_softmax(dim=1)
    return x 


tc_model = bobnet() 
to_model = torchnet()

tinychad_optim = SGD([tc_model.l1w, tc_model.l2w, tc_model.l1b, tc_model.l2b], lr=1e-3)
torch_optim = torch.optim.SGD([to_model.l1w, to_model.l2w, to_model.l1b, to_model.l2b], lr=1e-3)

N = np.random.randn(1,3).astype(np.float32)
x = tensor(N, requires_grad = True)
xt = torch.tensor(N, requires_grad = True)

# for LOSS
loss_fn = nn.CrossEntropyLoss()

R = np.array([[1,0,0]]).astype(np.float32)
R_tc = np.array([[1,0,0]]).astype(np.float32)

# steps:
for j in range(10):
  s = tc_model.forward(x)
  st = to_model.forward(xt)

  s_tinygrad = Tensor(s.data)

  np.testing.assert_allclose(s.data, st.detach().numpy(), atol =1e-6 , rtol =1e-3)

  st_l = loss_fn(st, torch.tensor(R))
  s_l = s.cross_entropy_loss(R_tc)

  print(st_l.detach().numpy(), s_l.data, s_tinygrad.numpy())

  tinychad_optim.zero_grad()
  torch_optim.zero_grad()

  st_l.backward()
  s_l.backward()

  tinychad_optim.step()
  torch_optim.step()

  np.testing.assert_allclose(tc_model.l2w.data, to_model.l2w.detach().numpy(), atol =1e-4 , rtol =1e-3)
  np.testing.assert_allclose(tc_model.l1w.data, to_model.l1w.detach().numpy(), atol =1e-4 , rtol =1e-3)
  np.testing.assert_allclose(tc_model.l2b.data, to_model.l2b.detach().numpy(), atol =1e-4 , rtol =1e-3)
  np.testing.assert_allclose(tc_model.l1b.data, to_model.l1b.detach().numpy(), atol =1e-4 , rtol =1e-3)













