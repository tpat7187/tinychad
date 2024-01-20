from tinychad.tensor import tensor
import torch
import numpy as np 
import unittest


def op_test_helper(shape, tinychadfxn, torchfxn, axis=None, keepdim=None):
  np.random.seed(0)
  UNARY_OPS = (tensor.relu, tensor.exp, tensor.log, tensor.neg, tensor.sqrt)
  BINARY_OPS = (tensor.add, tensor.sub, tensor.div, tensor.mul)
  SHAPE_OPS = (tensor.sum, tensor.max)

  if tinychadfxn in UNARY_OPS: src = 1
  if tinychadfxn in BINARY_OPS: src = 2
  if tinychadfxn in SHAPE_OPS: src = 1

  chi = [] 
  if not isinstance(shape, list):
    for _ in range(src):
      chi.append(np.random.randn(*shape).astype(np.float32))
  else: 
    for _ in range(src):
      chi.append(np.random.randn(*shape[_]).astype(np.float32))

  tinychad_tensors = [] 
  torch_tensors = []
  for i in chi: 
    tinychad_tensors.append(tensor(i))
    torch_tensors.append(torch.tensor(i))

  if tinychadfxn in UNARY_OPS:
    x = tinychadfxn(*tinychad_tensors)
    xt = torchfxn(*torch_tensors)
  if tinychadfxn in BINARY_OPS: 
    x = tinychadfxn(*tinychad_tensors)
    xt = torchfxn(*torch_tensors)
  if tinychadfxn == tensor.sum: 
    x = tinychadfxn(*tinychad_tensors, axis = axis) if axis is not None else tinychadfxn(*tinychad_tensors)
    xt = torchfxn(*torch_tensors, dim = axis) if axis is not None else torchfxn(*torch_tensors)
  if tinychadfxn == tensor.max:
    x = tinychadfxn(*tinychad_tensors, axis = axis) if axis is not None else tinychadfxn(*tinychad_tensors)
    xt = torchfxn(*torch_tensors,  dim = axis) if axis is not None else torchfxn(*torch_tensors)
    if axis is not None:
      xt = xt.values

  x.realize()
  try: 
    np.testing.assert_allclose(x.detach(), xt.numpy(), atol =1e-6 , rtol =1e-3)
  except Exception: 
    raise Exception(f"<fw pass failure> tinychad:\n {x.detach()} \n, pytorch:\n {xt.numpy()}")


class run_backend_tests(unittest.TestCase): 

  def test_add(self): return op_test_helper((5,5), tensor.add, torch.add) 
  def test_add_cast(self): return op_test_helper([(5,1),(5,5)], tensor.add, torch.add) 
  def test_add_reshape_cast(self): return op_test_helper([(5,1),(5,5,5)], tensor.add, torch.add) 

  def test_sub(self): return op_test_helper((5,5), tensor.sub, torch.sub) 
  def test_sub_cast(self): return op_test_helper([(5,1),(5,5)], tensor.add, torch.add) 
  def test_sub_reshape_cast(self): return op_test_helper([(5,1),(5,5,5)], tensor.add, torch.add) 

  def test_div(self): return op_test_helper((5,5), tensor.div, torch.div) 
  def test_div_cast(self): return op_test_helper([(5,1),(5,5)], tensor.add, torch.add) 
  def test_div_reshape_cast(self): return op_test_helper([(5,1),(5,5,5)], tensor.add, torch.add) 

  def test_mul(self): return op_test_helper((5,5), tensor.mul, torch.mul) 
  def test_mul_cast(self): return op_test_helper([(5,1),(5,5)], tensor.add, torch.add) 
  def test_mul_reshape_cast(self): return op_test_helper([(5,1),(5,5,5)], tensor.add, torch.add) 

  def test_exp(self): return op_test_helper((5,5), tensor.exp, torch.exp)
  def test_log(self): return op_test_helper((5,5), tensor.log, torch.log)
  def test_sqrt(self): return op_test_helper((5,5), tensor.sqrt, torch.sqrt)
  def test_relu(self): return op_test_helper((5,5), tensor.relu, torch.relu)

  def test_sum(self): 
    op_test_helper((5,5), tensor.sum, torch.sum)
    op_test_helper((5,5,5), tensor.sum, torch.sum, axis = 0)
    op_test_helper((5,3,5), tensor.sum, torch.sum, axis = 1)
    op_test_helper((5,3,4,1,5), tensor.sum, torch.sum, axis = 2)

  def test_max(self): 
    op_test_helper((5,5), tensor.max, torch.max)
    op_test_helper((5,5,5), tensor.max, torch.max, axis = 0)
    op_test_helper((5,3,5), tensor.max, torch.max, axis = 1)
    op_test_helper((5,3,4,1,5), tensor.max, torch.max, axis = 2)

if __name__ == "__main__": 
  unittest.main()







