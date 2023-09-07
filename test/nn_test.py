import sys
from tinychad.tensor import tensor, Linear, Conv2d, BatchNorm2d
from tinychad.optim import SGD
from tinychad.helpers import get_parameters
import numpy as np
import torch
import torch.nn as nn
import unittest

def test_linear_helper(input_size, output_size, steps, batch_size):
  np.random.seed(5)
  IN = np.random.randn(batch_size, input_size).astype(np.float32)
  W = np.random.randn(input_size, output_size).astype(np.float32)
  B = np.random.randn(output_size).astype(np.float32)

  tc_layer = Linear(input_size, output_size)
  tc_layer.w = tensor(W, requires_grad=True)
  tc_layer.b = tensor(B, requires_grad=True)

  torch_layer = torch.nn.Linear(input_size, output_size)
  torch_layer.weight.data = torch.tensor(W.T, requires_grad=True)
  torch_layer.bias.data = torch.tensor(B, requires_grad=True)

  tc_param = get_parameters(tc_layer)
  tinychad_optim = SGD(tc_param, lr=1e-3, momentum=0.2)
  torch_optim = torch.optim.SGD([torch_layer.weight, torch_layer.bias], lr=1e-3, momentum=0.2)

  torch_in = torch.tensor(IN)
  tc_in = tensor(IN)

  real = np.random.randint(0, output_size, (batch_size,))
  for j in range(steps):
    tc_out = tc_layer(tc_in)
    torch_out = torch_layer(torch_in)

    tc_real = tensor(real)
    torch_real = torch.tensor(real)

    loss_fn = nn.NLLLoss()

    torch_loss = loss_fn(torch_out, torch_real)
    tc_loss = tc_out.NLLLoss(tc_real)

    tinychad_optim.zero_grad()
    torch_optim.zero_grad()

    torch_loss.backward()
    tc_loss.backward()

    tinychad_optim.step()
    torch_optim.step()

    np.testing.assert_allclose(torch_loss.item(), tc_loss.data.dat, atol=1e-5, rtol=1e-3)


  
def test_conv2d_helper(in_channels, out_channels, kernel_size, steps, batch_size, padding, stride):
  kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
  IN = np.random.randn(batch_size, in_channels, 10,10).astype(np.float32)

  W = np.random.randn(out_channels, in_channels, *kernel_size).astype(np.float32)
  B = np.random.randn(out_channels).astype(np.float32)

  Wfc = np.random.randn(5*5*5, 10).astype(np.float32)
  Bfc = np.random.randn(10).astype(np.float32)


  tc_layer = Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
  tc_layer.w = tensor(W, requires_grad=True)
  tc_layer.b = tensor(B, requires_grad=True)

  tc_fc_layer = Linear(5*5*5, 10)
  tc_fc_layer.w = tensor(Wfc, requires_grad=True)
  tc_fc_layer.b = tensor(Bfc, requires_grad=True)

  torch_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
  torch_layer.weight.data = torch.tensor(W, requires_grad=True)
  torch_layer.bias.data = torch.tensor(B, requires_grad=True)

  torch_fc_layer = nn.Linear(3*5*5*5, 10)
  torch_fc_layer.weight.data = torch.tensor(Wfc.T, requires_grad=True)
  torch_fc_layer.bias.data = torch.tensor(Bfc, requires_grad=True)

  tinychad_optim = SGD([tc_layer.w, tc_layer.b, tc_fc_layer.w, tc_fc_layer.b], lr=1e-3)
  torch_optim = torch.optim.SGD([torch_layer.weight, torch_layer.bias, torch_fc_layer.weight, torch_fc_layer.bias], lr=1e-3)

  torch_in = torch.tensor(IN)
  tc_in = tensor(IN)

  real = np.random.randint(0, 10, (batch_size,))
  for j in range(steps):
    tc_out = tc_layer(tc_in).reshape(batch_size,-1)
    torch_out = torch_layer(torch_in).reshape(batch_size,-1)

    tc_outLinear = tc_fc_layer(tc_out)
    torch_outLinear= torch_fc_layer(torch_out)
    

    tc_real = tensor(real)
    torch_real = torch.tensor(real)

    loss_fn = nn.NLLLoss()

    torch_loss = loss_fn(torch_outLinear, torch_real)
    tc_loss = tc_outLinear.NLLLoss(tc_real)

    tinychad_optim.zero_grad()
    torch_optim.zero_grad()

    torch_loss.backward()
    tc_loss.backward()

    tinychad_optim.step()
    torch_optim.step()

    np.testing.assert_allclose(torch_loss.item(), tc_loss.data.dat, atol=1e-5, rtol=1e-3)


def test_batchnorm2d_helper(self, steps):
  pass

class test_nn(unittest.TestCase):
  def test_linear_nn(self): return test_linear_helper(10, 5, steps=10, batch_size=3)
  def test_linear_nn(self): return test_linear_helper(15, 10, steps=10, batch_size=10)
  def test_linear_nn(self): return test_linear_helper(28, 3, steps=100, batch_size=3)

  def test_conv2d_nn(self): return test_conv2d_helper(3,5,2, padding=0, stride=2, batch_size=3, steps=50)


if __name__ == "__main__": 
  unittest.main()


