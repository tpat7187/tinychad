import numpy as np 
import os

class OP: 
  def __init__(self, saved = None, ctx = None):
    self.arg = type(self).__name__
    self.saved = np.array(saved)
    self.ctx = ctx

import tinychad.ops as ops

#### TENSOR CLASS ####
class tensor: 
  def __init__(self, data, op = ops.LOAD(), requires_grad = False):
    self.data, self.op = np.array(data, dtype = np.float32), op
    self.grad = np.zeros(self.data.shape, dtype = np.float32)

  def ones(*shape): return tensor(np.ones(*shape))
  def randn(*shape): return tensor(np.random.randn(*shape))
  def eye(shape): return tensor(np.eye(shape))
  def zeros(*shape): return tensor(np.zeros(*shape))

  @property
  def shape(self): return self.data.shape

  @property
  def dtype(self): return self.data.dtype

  @property
  def size(self): return self.data.size

  def __repr__(self): 
    return f"op = <{self.op.arg}>: shape = {self.data.shape}: grad_shape = {self.grad.shape}"

  def __add__(self,x): return self.add(x)
  def __sub__(self,x): return self.sub(x)
  def __mul__(self,x): return self.mul(x)
  def __matmul__(self,x): return self.dot(x)
  def __truediv__(self,x): return self.div(x)

  def __radd__(self,x): return self.add(x)
  def __rsub__(self,x): return self.sub(x)
  def __rmul__(self,x): return self.mul(x)

  # binary 
  #def add(self, x): return tensor(ops.ADD.forward(self, x), op = ops.ADD(saved = [self, x]))
  def add(self, x): return self.cast_op(ops.ADD, x) 
  def sub(self, x): return self.cast_op(ops.SUB, x) 
  def mul(self, x): return self.cast_op(ops.MUL, x) 
  def div(self, x): return self.cast_op(ops.DIV, x)

  def dot(self, x): return tensor(ops.MATMUL.forward(self, x), op = ops.MATMUL(saved = [self,x]))

  # unary
  def sum(self, axis = None, keepdim = False): return tensor(ops.SUM.forward(self, axis, keepdim), op = ops.SUM(saved = [self,], ctx=axis))
  def relu(self): return tensor(ops.RELU.forward(self), op = ops.RELU(saved = [self,]))
  def exp(self): return tensor(ops.EXP.forward(self), op = ops.EXP(saved = [self,]))
  def log(self): return tensor(ops.LOG.forward(self), op = ops.LOG(saved = [self,]))
  def reshape(self, *shape) : return tensor(ops.RESHAPE.forward(self, *shape), op = ops.RESHAPE(saved = [self,]))
  def max(self, axis = None, keepdim = False): return tensor(ops.MAX.forward(self, axis, keepdim), op = ops.MAX(saved = [self,], ctx=axis))

  def cast(self, x, ctx): return tensor(ops.CAST.forward(self, x), op = ops.CAST(saved = [self,], ctx = ctx))

  # helpers
  def T(self): return tensor(self.data.transpose())
  def argmax(self, axis = None): return self.data.argmax(axis=axis)

  # from tinygrad
  def _softmax(self, axis): 
    m = self - self.max(axis=axis, keepdim=True)
    e = m.exp() 
    return m, e, e.sum(axis=axis, keepdim=True)

  # mlops
  def softmax(self, axis = -1):
    m = self - self.max(axis = axis, keepdim=True)
    e = m.exp()
    ss = e.sum(axis=axis, keepdim=True)
    return e.div(ss)
 
  #they both give correct output, something is wrong with the other ops, we should write tests
  def logsoftmax(self, axis=-1):
    m, _, ss = self._softmax(axis) 
    return m - ss.log()

  def toposort(self): 
    topo, vis = [], []
    def _toposort(s): 
      if s not in vis: 
        vis.append(s)
        if not isinstance(s.op, ops.LOAD):
          for child in s.op.saved: 
            _toposort(child)
          topo.append(s)
    _toposort(self)
    
    # should we include load ops
    return topo

  def backward(self):
    DEBUG = os.getenv("DEBUG") 

    assert(self.grad.shape == (1,))
    self.grad = np.ones(self.grad.shape)
    for x in reversed(self.toposort()): 
      assert x.grad.shape == x.shape, \
        f"grad shape must match tensor shape in {x.grad.shape} != {x.shape} on {x.op.arg}"
      if DEBUG == "1":
        in_s = list(n.shape for n in x.op.saved)
        print(f"op = <{x.op.arg}> in: {in_s} -> out: {x.data.shape} with grad: {x.grad.shape}")
      x.op.backward(x.grad, x.data)

  def cast_op(self, fxn, x):
    x, y = self, x 
    if x.shape == y.shape: 
      return tensor(fxn.forward(x,y), op = fxn(saved = [x, y]))
    cst, shp, ot, axis = castable(x,y)
    cst = cst.cast(shp, ctx = shp)
    return tensor(fxn.forward(cst, ot), op = fxn(saved = [cst, ot]))

# returns cast and output shape, sum axis idk if we need
# castable checks cast decision returns cast target and shape target
def castable(x, y): 
  assert is_castable(x,y), f"shapes {x.shape} and {y.shape} are not castable"
  out = np.broadcast_shapes(x.shape, y.shape)
  if x.shape != out:
    return x, out, y, 0 
  if y.shape != out:
    return y, out, x, 0

def is_castable(x, y): 
  for a, b in zip(x.shape[::-1], y.shape[::-1]): 
    if a == 1 or b == 1 or a == b:
      pass
    else: 
      return False
  return True





  





