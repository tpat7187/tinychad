import sys

sys.path.insert(1, '../')

from tinychad.tensor import tensor
import numpy as np
import torch
import unittest

# TODO: input shapes, reshape tests, casting tests
def test_helper_fw(shapes, torchfxn, tinychadfxn, axis = None):
    np.random.seed(0)
    N = np.random.randn(5,5).astype(np.float32)

    a, b = tensor(N, requires_grad = True), tensor(N, requires_grad = True)
    at, bt = torch.tensor(N, requires_grad = True), torch.tensor(N, requires_grad = True)


    UNARY_OPS = (tensor.sum, tensor.relu, tensor.exp, tensor.log, tensor.max, tensor.neg)
    BINARY_OPS = (tensor.add, tensor.sub, tensor.div, tensor.mul, tensor.dot, tensor.cast)
    RESHAPE_OPS = (tensor.reshape, None)
    CAST_OPS = (tensor.cast, None)

    if tinychadfxn in UNARY_OPS:
      x = tinychadfxn(a, axis = shapes) if axis != None else tinychadfxn(a)
      xt = torchfxn(at, axis = shapes) if axis != None else torchfxn(at)
    if tinychadfxn in BINARY_OPS:
      x = tinychadfxn(a,b)
      xt = torchfxn(at,bt)


    if tinychadfxn in RESHAPE_OPS:
      x = tinychadfxn(a, *shapes)
      xt = torchfxn(at, (shapes))

    if tinychadfxn in CAST_OPS: 
      x = tinychadfxn(a, ctx = (shapes))
      xt = tinychadfxn(a, (shapes))

    try: 
      np.testing.assert_allclose(x.data, xt.detach().numpy(), atol =1e-6 , rtol =1e-3)
    except Exception: 
      raise Exception(f"<fw pass failure> tinychad: {x.data}, pytorch{xt.detatch().numpy()}")

def test_helper_bw(shapes, torchfxn, tinychadfxn, axis = None):
    np.random.seed(0)
    N = np.random.randn(5,5).astype(np.float32)

    a, b = tensor(N, requires_grad = True), tensor(N, requires_grad = True)
    at, bt = torch.tensor(N, requires_grad = True), torch.tensor(N, requires_grad = True)

    UNARY_OPS = (tensor.sum, tensor.relu, tensor.exp, tensor.log, tensor.max, tensor.neg)
    BINARY_OPS = (tensor.add, tensor.sub, tensor.div, tensor.mul, tensor.dot)
    RESHAPE_OPS = (tensor.reshape, tensor.cast)

    if tinychadfxn in UNARY_OPS:
      x = tinychadfxn(a, axis = axis).sum().backward() if axis != None else tinychadfxn(a).sum().backward()
      xt = torchfxn(at, axis = axis).sum().backward() if axis != None else torchfxn(at).sum().backward()

    if tinychadfxn in BINARY_OPS:
      x = tinychadfxn(a,b).sum().backward()
      xt = torchfxn(at, bt).sum().backward()

    if tinychadfxn in RESHAPE_OPS:
      x = tinychadfxn(a, *shapes).sum().backward()
      xt = torchfxn(at, (shapes)).sum().backward()


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

  def test_relu_fw(self): return test_helper_fw(None, torch.relu, tensor.relu)
  def test_relu_bw(self): return test_helper_bw(None, torch.relu, tensor.relu)

  def test_sum_fw(self): return test_helper_fw(None, torch.sum, tensor.sum)
  def test_sum_bw(self): return test_helper_bw(None, torch.sum, tensor.sum)

  def test_exp_fw(self): return test_helper_fw(None, torch.exp, tensor.exp)
  def test_exp_bw(self): return test_helper_bw(None, torch.exp, tensor.exp)

  def test_log_fw(self): return test_helper_fw(None, torch.log, tensor.log)
  def test_log_bw(self): return test_helper_bw(None, torch.log, tensor.log)

  def test_max_fw(self): return test_helper_fw(None, torch.max, tensor.max)
  def test_max_bw(self): return test_helper_bw(None, torch.max, tensor.max)

  def test_neg_fw(self): return test_helper_fw(None, torch.neg, tensor.neg)
  def test_neg_bw(self): return test_helper_bw(None, torch.neg, tensor.neg)

  def test_reshape_fw(self): return test_helper_fw((-1,1), torch.reshape, tensor.reshape)
  def test_reshape_bw(self): return test_helper_bw((-1,1), torch.reshape, tensor.reshape)

  def test_sum0_fw(self): return test_helper_fw(1, torch.sum, tensor.sum, axis=0)
  def test_sum0_bw(self): return test_helper_bw(1, torch.sum, tensor.sum, axis=0)
  def test_sum1_fw(self): return test_helper_fw(1, torch.sum, tensor.sum, axis=1)
  def test_sum1_bw(self): return test_helper_bw(1, torch.sum, tensor.sum, axis=1)

  # how to test this, torch.max for axis != None does not return a tensor
  '''
  def test_max0_fw(self): return test_helper_fw(1, torch.max, tensor.max, axis=0)
  def test_max0_bw(self): return test_helper_bw(1, torch.max, tensor.max, axis=0)
  def test_max1_fw(self): return test_helper_fw(1, torch.max, tensor.max, axis=1)
  def test_max1_bw(self): return test_helper_bw(1, torch.max, tensor.max, axis=1)

  TRY TO REMOVE NEED OF A CONTEXT
  def test_cast_fw(self): return test_helper_fw((5,5,5), torch.broadcast_to, tensor.cast)
  def test_cast_bw(self): return test_helper_bw((5,5,5), torch.broadcast_to, tensor.cast)
  '''


if __name__ == "__main__": 
  unittest.main()





