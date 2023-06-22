import sys

sys.path.insert(1, '../')

from tinychad.tensor import tensor
import numpy as np
import torch
import unittest


# TODO: input shapes, reshape tests, casting tests
def test_helper_fw(shapes, torchfxn, tinychadfxn):
    np.random.seed(0)
    N = np.random.randn(5,5).astype(np.float32)

    a, b = tensor(N, requires_grad = True), tensor(N, requires_grad = True)
    at, bt = torch.tensor(N, requires_grad = True), torch.tensor(N, requires_grad = True)


    UNARY_OPS = (tensor.sum, tensor.relu, tensor.exp, tensor.log, tensor.max, tensor.reshape, tensor.neg)
    BINARY_OPS = (tensor.add, tensor.sub, tensor.div, tensor.mul, tensor.dot)
    RESHAPE_OPS = (tensor.reshape, tensor.cast)

    if tinychadfxn in UNARY_OPS:
      x = tinychadfxn(a)
      xt = torchfxn(at)
    if tinychadfxn in BINARY_OPS:
      x = tinychadfxn(a,b)
      xt = torchfxn(at,bt)

    try: 
      np.testing.assert_allclose(x.data, xt.detach().numpy(), atol =1e-6 , rtol =1e-3)
    except Exception: 
      raise Exception(f"<fw pass failure> tinychad: {x.data}, pytorch{xt.detatch().numpy()}")

def test_helper_bw(shapes, torchfxn, tinychadfxn):
    N = np.random.randn(5,5).astype(np.float32)

    a, b = tensor(N, requires_grad = True), tensor(N, requires_grad = True)
    at, bt = torch.tensor(N, requires_grad = True), torch.tensor(N, requires_grad = True)

    UNARY_OPS = (tensor.sum, tensor.relu, tensor.exp, tensor.log, tensor.max, tensor.reshape, tensor.neg)
    BINARY_OPS = (tensor.add, tensor.sub, tensor.div, tensor.mul, tensor.dot)
    RESHAPE_OPS = (tensor.reshape, tensor.cast)

    if tinychadfxn in UNARY_OPS:
      x = tinychadfxn(a).sum().backward()
      xt = torchfxn(at).sum().backward()

    if tinychadfxn in BINARY_OPS:
      x = tinychadfxn(a,b).sum().backward()
      xt = torchfxn(at, bt).sum().backward()

    try: 
      np.testing.assert_allclose(a.grad, at.grad, atol=1e-6, rtol=1e-3)
    except Exception: 
      raise Exception(f"<bw pass failure> tinychad: {a.grad} \n pytorch{at.grad}")

    if tinychadfxn in BINARY_OPS: 
      try:
        np.testing.assert_allclose(b.grad, bt.grad, atol=1e-6, rtol=1e-3)
      except Exception: 
        raise Exception(f"<bw pass failure> tinychad: {b.grad} \n pytorch{bt.grad}")


class test_ops(unittest.TestCase): 

  def test_add_fw(self): return test_helper_fw(1, torch.add, tensor.add)
  def test_add_bw(self): return test_helper_bw(1, torch.add, tensor.add)

  def test_sub_fw(self): return test_helper_fw(1, torch.sub, tensor.sub)
  def test_sub_bw(self): return test_helper_bw(1, torch.sub, tensor.sub)

  def test_mul_fw(self): return test_helper_fw(1, torch.mul, tensor.mul)
  def test_mul_bw(self): return test_helper_bw(1, torch.mul, tensor.mul)

  def test_div_fw(self): return test_helper_fw(1, torch.div, tensor.div)
  def test_div_bw(self): return test_helper_bw(1, torch.div, tensor.div)
   
  def test_matmul_fw(self): return test_helper_fw(1, torch.matmul, tensor.dot)
  def test_matmul_bw(self): return test_helper_bw(1, torch.matmul, tensor.dot)

  def test_relu_fw(self): return test_helper_fw(1, torch.relu, tensor.relu)
  def test_relu_bw(self): return test_helper_bw(1, torch.relu, tensor.relu)

  def test_sum_fw(self): return test_helper_fw(1, torch.sum, tensor.sum)
  def test_sum_bw(self): return test_helper_bw(1, torch.sum, tensor.sum)

  def test_exp_fw(self): return test_helper_fw(1, torch.exp, tensor.exp)
  def test_exp_bw(self): return test_helper_bw(1, torch.exp, tensor.exp)

  def test_log_fw(self): return test_helper_fw(1, torch.log, tensor.log)
  def test_log_bw(self): return test_helper_bw(1, torch.log, tensor.log)

  def test_max_fw(self): return test_helper_fw(1, torch.max, tensor.max)
  def test_max_bw(self): return test_helper_bw(1, torch.max, tensor.max)

  def test_neg_fw(self): return test_helper_fw(1, torch.neg, tensor.neg)
  def test_neg_bw(self): return test_helper_bw(1, torch.neg, tensor.neg)







if __name__ == "__main__": 
  unittest.main()



