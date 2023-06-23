import sys

sys.path.insert(1, '../')

from tinychad.tensor import tensor
import numpy as np
import torch
import unittest


def test_MNIST_fw():
  np.random.seed(0)
  BS = 69

  inp = np.random.randn(BS,28,28).astype(np.float32)
  res = np.random.randn(BS,10)
  
  W1 = np.random.randn(28*28,128).astype(np.float32)
  W2 = np.random.randn(128,10).astype(np.float32)
  B1 = np.random.randn(128).astype(np.float32)
  B2 = np.random.randn(10).astype(np.float32)

  # tinychad
  inp_tc = tensor(inp).reshape(BS,-1)
  res_tc = tensor(res)
  
  w1_tc = tensor(W1)
  w2_tc = tensor(W2)
  b1_tc = tensor(B1)
  b2_tc = tensor(B2)

  # fw pass
  l1_tc = (inp_tc.dot(w1_tc) + b1_tc).relu()
  l2_tc = (l1_tc.dot(w2_tc) + b2_tc).relu()
  l3_tc = l2_tc.logsoftmax(axis=1)

  loss_tc = (res_tc * (l3_tc)).mean()

  # torch
  inp_to = torch.tensor(inp).reshape(BS,-1)
  res_to = torch.tensor(res)

  w1_to = torch.tensor(W1, requires_grad = True)
  w2_to = torch.tensor(W2, requires_grad = True)
  b1_to = torch.tensor(B1, requires_grad = True)
  b2_to = torch.tensor(B2, requires_grad = True)

  # fw pass
  l1_to = (inp_to.matmul(w1_to) + b1_to).relu()
  l2_to = (l1_to.matmul(w2_to) + b2_to).relu()
  l3_to = l2_to.log_softmax(axis=1)

  loss_to = (res_to * (l3_to)).mean()

  loss_to.backward()
  loss_tc.backward()

  try: 
    np.testing.assert_allclose(w1_to.grad, w1_tc.grad, atol=1e-5, rtol=1e-3)
    np.testing.assert_allclose(w2_to.grad, w2_tc.grad, atol=1e-5, rtol=1e-3)
    np.testing.assert_allclose(b1_to.grad, b1_tc.grad, atol=1e-5, rtol=1e-3)
    np.testing.assert_allclose(b2_to.grad, b2_tc.grad, atol=1e-5, rtol=1e-3)
  except Exception: 
    raise Exception('error in MNIST test backward')


class test_mnist(unittest.TestCase): 
  def test_mnist(self): return test_MNIST_fw()


if __name__ == "__main__": 
  unittest.main()





