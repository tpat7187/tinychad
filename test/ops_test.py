import sys

sys.path.insert(1, '../')

from tinychad.tensor import tensor
import numpy as np
import torch
import unittest


# TODO: specify input shapes, casting tests
def test_helper_fw(shapes, torchfxn, tinychadfxn, axis = None):
    np.random.seed(0)
    N = np.random.randn(5,5).astype(np.float32)

    a, b = tensor(N, requires_grad = True), tensor(N, requires_grad = True)
    at, bt = torch.tensor(N, requires_grad = True), torch.tensor(N, requires_grad = True)

    UNARY_OPS = (tensor.sum, tensor.relu, tensor.exp, tensor.log, tensor.neg, tensor.mean)
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
    if tinychadfxn == tensor.max:
      x = tinychadfxn(a, axis = axis) if axis != None else tinychadfxn(a)
      xt = torchfxn(at, dim = axis).values if axis != None else torchfxn(at)

    try: 
      np.testing.assert_allclose(x.exec().data, xt.detach().numpy(), atol =1e-6 , rtol =1e-3)
    except Exception: 
      raise Exception(f"<fw pass failure> tinychad: {x.data}, pytorch{xt.detatch().numpy()}")

def test_helper_bw(shapes, torchfxn, tinychadfxn, axis = None):
    np.random.seed(0)
    N = np.random.randn(5,5).astype(np.float32)

    a, b = tensor(N, requires_grad = True), tensor(N, requires_grad = True)
    at, bt = torch.tensor(N, requires_grad = True), torch.tensor(N, requires_grad = True)

    UNARY_OPS = (tensor.sum, tensor.relu, tensor.exp, tensor.log, tensor.neg, tensor.mean)
    BINARY_OPS = (tensor.add, tensor.sub, tensor.div, tensor.mul, tensor.dot)
    RESHAPE_OPS = (tensor.reshape, tensor.cast)
    CAST_OPS = (tensor.cast, None)


    if tinychadfxn in UNARY_OPS:
      x = tinychadfxn(a, axis = axis).sum().backward() if axis != None else tinychadfxn(a).sum().backward()
      xt = torchfxn(at, axis = axis).sum().backward() if axis != None else torchfxn(at).sum().backward()
    if tinychadfxn in BINARY_OPS:
      x = tinychadfxn(a,b).sum().backward()
      xt = torchfxn(at, bt).sum().backward()
    if tinychadfxn in RESHAPE_OPS:
      x = tinychadfxn(a, *shapes).sum().backward()
      xt = torchfxn(at, (shapes)).sum().backward()
    if tinychadfxn == tensor.max:
      x = tinychadfxn(a, axis = axis).sum().backward() if axis != None else tinychadfxn(a).sum().backward()
      xt = torchfxn(at, dim = axis).values.sum().backward() if axis != None else torchfxn(at).sum().backward()

    try: 
      np.testing.assert_allclose(a.grad, at.grad, atol=1e-6, rtol=1e-3)
    except Exception: 
      raise Exception(f"<bw pass failure> tinychad: {a.grad} \n pytorch{at.grad}")

    if tinychadfxn in BINARY_OPS: 
      try:
        np.testing.assert_allclose(b.grad, bt.grad, atol=1e-6, rtol=1e-3)
      except Exception: 
        raise Exception(f"<bw pass failure> tinychad: {b.grad} \n pytorch{bt.grad}")

def conv_pool_test_helper_fw(tinychadfxn, torchfxn, input_shape, kernel_size, bias = None, padding=0, stride=1): 
  np.random.seed(0)

  N = np.random.randn(*input_shape).astype(np.float32)
  if tinychadfxn == tensor.conv2d and bias != None:
    W = np.random.randn(bias).astype(np.float32)
    c = tensor(W, requires_grad = True)
    ct = torch.tensor(W, requires_grad = True)
  else:
    c, ct = None, None
  K = np.random.randn(1,input_shape[1],*kernel_size).astype(np.float32)

  a, b = tensor(N, requires_grad = True), tensor(K, requires_grad = True)
  at, bt = torch.tensor(N, requires_grad = True), torch.tensor(K, requires_grad = True)

  FXN = [tensor.conv2d, tensor.max_pool2d, tensor.avg_pool2d]

  if tinychadfxn == tensor.conv2d:
    x = tinychadfxn(a, weight=b, bias=c, padding=padding, stride=stride)
    xt = torchfxn(at, weight=bt, bias=ct, padding=padding, stride=stride)
    y = x.sum().backward()
    yt = xt.sum().backward()
  if tinychadfxn == tensor.avg_pool2d:
    x = tinychadfxn(a, kernel_size, stride=stride)
    xt = torchfxn(at, kernel_size, stride=stride)
    y = x.sum().backward()
    yt = xt.sum().backward()
  if tinychadfxn == tensor.max_pool2d:
    x = tinychadfxn(a, kernel_size=kernel_size[0], stride=kernel_size[0])
    xt = torchfxn(at, kernel_size=kernel_size[0], stride=kernel_size[0])
    y = x.sum().backward()
    yt = xt.sum().backward()

  try: 
    np.testing.assert_allclose(x.data, xt.detach().numpy(), atol=1e-6, rtol=1e-3)
    np.testing.assert_allclose(a.grad, at.grad, atol=1e-6, rtol=1e-3)
    if tinychadfxn == tensor.conv2d and bias != None:
      np.testing.assert_allclose(b.grad, bt.grad, atol=1e-6, rtol=1e-3)
      np.testing.assert_allclose(c.grad, ct.grad, atol=1e-6, rtol=1e-3)

  except Exception: 
    raise Exception(f"<fw pass failure> tinychad: {x.data} \n pytorch{xt}")


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

  def test_sum_fw(self): 
    test_helper_fw(None, torch.sum, tensor.sum)
    test_helper_fw(None, torch.sum, tensor.sum, axis = 0)
    test_helper_fw(None, torch.sum, tensor.sum, axis = 1)

  def test_sum_bw(self): 
    test_helper_bw(None, torch.sum, tensor.sum)
    test_helper_bw(None, torch.sum, tensor.sum, axis = 0)
    test_helper_bw(None, torch.sum, tensor.sum, axis = 1)

  def test_exp_fw(self): return test_helper_fw(None, torch.exp, tensor.exp)
  def test_exp_bw(self): return test_helper_bw(None, torch.exp, tensor.exp)

  def test_log_fw(self): return test_helper_fw(None, torch.log, tensor.log)
  def test_log_bw(self): return test_helper_bw(None, torch.log, tensor.log)

  def test_neg_fw(self): return test_helper_fw(None, torch.neg, tensor.neg)
  def test_neg_bw(self): return test_helper_bw(None, torch.neg, tensor.neg)

  def test_reshape_fw(self): return test_helper_fw((-1,1), torch.reshape, tensor.reshape)
  def test_reshape_bw(self): return test_helper_bw((-1,1), torch.reshape, tensor.reshape)

  def test_mean_fw(self): 
    test_helper_fw(None, torch.mean, tensor.mean)
    test_helper_fw(None, torch.mean, tensor.mean, axis=0)
    test_helper_fw(None, torch.mean, tensor.mean, axis=1)

  def test_mean_bw(self):
    test_helper_bw(None, torch.mean, tensor.mean)
    test_helper_bw(None, torch.mean, tensor.mean, axis=0)
    test_helper_bw(None, torch.mean, tensor.mean, axis=1)

  def test_max_fw(self): 
    test_helper_fw(None, torch.max, tensor.max)
    test_helper_fw(None, torch.max, tensor.max, axis = 0)
    test_helper_fw(None, torch.max, tensor.max, axis = 1)

  def test_max_bw(self):
    test_helper_bw(None, torch.max, tensor.max)
    test_helper_bw(None, torch.max, tensor.max, axis = 0)
    test_helper_bw(None, torch.max, tensor.max, axis = 1)

  def test_conv_fw(self): 
    conv_pool_test_helper_fw(tensor.conv2d, torch.conv2d, (1,1,26,26), (3,3), padding=0, bias=1)
    conv_pool_test_helper_fw(tensor.conv2d, torch.conv2d, (1,3,26,26), (5,5), padding=0, bias=1)
    conv_pool_test_helper_fw(tensor.conv2d, torch.conv2d, (3,3,26,26), (3,3), padding=0, bias=1)
    conv_pool_test_helper_fw(tensor.conv2d, torch.conv2d, (3,3,26,26), (3,3), padding=2, stride=1, bias=1)
    conv_pool_test_helper_fw(tensor.conv2d, torch.conv2d, (3,3,26,26), (5,3), padding=2, stride=2, bias=1)

  def test_avg_pool_fw(self): 
    conv_pool_test_helper_fw(tensor.avg_pool2d, torch.nn.functional.avg_pool2d, (1,1,26,26), (3,3))
    conv_pool_test_helper_fw(tensor.avg_pool2d, torch.nn.functional.avg_pool2d, (1,3,5,5), (3,3))
    conv_pool_test_helper_fw(tensor.avg_pool2d, torch.nn.functional.avg_pool2d, (3,3,26,26), (5,5), stride=1)
    conv_pool_test_helper_fw(tensor.avg_pool2d, torch.nn.functional.avg_pool2d, (3,2,26,26), (3,3), stride=2)
    conv_pool_test_helper_fw(tensor.avg_pool2d, torch.nn.functional.avg_pool2d, (5,1,26,26), (3,3), stride=3)

  def test_max_pool_fw(self): 
    conv_pool_test_helper_fw(tensor.max_pool2d, torch.nn.functional.max_pool2d, (1,1,6,6), (2,2))
    conv_pool_test_helper_fw(tensor.max_pool2d, torch.nn.functional.max_pool2d, (1,3,6,6), (3,3))
    conv_pool_test_helper_fw(tensor.max_pool2d, torch.nn.functional.max_pool2d, (3,3,24,24), (3,3))
    conv_pool_test_helper_fw(tensor.max_pool2d, torch.nn.functional.max_pool2d, (3,2,24,24), (3,3))
    conv_pool_test_helper_fw(tensor.max_pool2d, torch.nn.functional.max_pool2d, (5,1,24,24), (3,3))

if __name__ == "__main__": 
  unittest.main()





