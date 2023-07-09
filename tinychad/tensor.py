import numpy as np 
import os

class OP: 
  def __init__(self, saved = None, ctx = None):
    self.arg = type(self).__name__
    self.saved = np.array(saved)
    self.ctx = ctx

  def forward(x, y): return f"forward not implemented for {self.arg}" 
  def backward(self, out_grad, out): return f"backward not implemented for {self.arg}" 

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

  def __getitem__(self, args): return self.slice(args)

  def __add__(self,x): return self.add(x)
  def __sub__(self,x): return self.sub(x)
  def __mul__(self,x): return self.mul(x)
  def __matmul__(self,x): return self.dot(x)
  def __truediv__(self,x): return self.div(x)
  def __neg__(self): return self.neg()

  def __radd__(self,x): return self.add(x)
  def __rsub__(self,x): return self.sub(x).neg()
  def __rmul__(self,x): return self.mul(x)

  def __iadd__(self,x): return self.add(x)
  def __isub__(self,x): return self.sub(x)
  def __imul__(self,x): return self.mul(x)

  # binary ops
  def add(self, x): return self.cast_op(ops.ADD, x) 
  def sub(self, x): return self.cast_op(ops.SUB, x) 
  def mul(self, x): return self.cast_op(ops.MUL, x) 
  def div(self, x): return self.cast_op(ops.DIV, x)

  def cast(self, x, ctx): return tensor(ops.CAST.forward(self, x), op = ops.CAST(saved = [self,], ctx = ctx))

  # MATMUL
  def dot(self, x): return tensor(ops.MATMUL.forward(self, x), op = ops.MATMUL(saved = [self,x]))

  # unary ops
  def relu(self): return tensor(ops.RELU.forward(self), op = ops.RELU(saved = [self,]))
  def exp(self): return tensor(ops.EXP.forward(self), op = ops.EXP(saved = [self,]))
  def log(self): return tensor(ops.LOG.forward(self), op = ops.LOG(saved = [self,]))
  def neg(self): return tensor(ops.NEG.forward(self), op = ops.NEG(saved = [self,]))

  # shape ops (changes shape and content)
  def max(self, axis = None, keepdim = False): return tensor(ops.MAX.forward(self, axis, keepdim), op = ops.MAX(saved = [self,], ctx=[axis, keepdim]))
  def sum(self, axis = None, keepdim = False): return tensor(ops.SUM.forward(self, axis, keepdim), op = ops.SUM(saved = [self,], ctx=axis))

  # reshape ops (changes shape, content does not change, sparse -> circular matrix for conv)
  def reshape(self, *shape) : return tensor(ops.RESHAPE.forward(self, *shape), op = ops.RESHAPE(saved = [self,]))
  def slice(self, *args) : return tensor(ops.SLICE.forward(self, *args), op = ops.SLICE(saved = [self,], ctx = args))
  def pad(self, *args, axis): return tensor(ops.PAD.forward(self, *args, axis), op = ops.PAD(saved = [self,], ctx = [args, axis]))
  def sparse(self, *shape) : return tensor(ops.SPARSE.forward(self, *shape), op = ops.SPARSE(saved = [self,]))

  # helpers
  def T(self): return tensor(self.data.transpose())
  def argmax(self, axis = None): return self.data.argmax(axis=axis)

  def mean(self, axis=None, keepdim=False): 
    out = self.sum(axis=axis, keepdim=keepdim)
    ss = out * (np.prod(out.shape) / np.prod(self.shape))
    return ss

  def _softmax(self, axis): 
    m = self - self.max(axis=axis, keepdim=True)
    e = m.exp() 
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis = -1):
    _, e, ss = self._softmax(axis) 
    return e.div(ss)
 
  def logsoftmax(self, axis=-1):
    m, _, ss = self._softmax(axis) 
    return m - ss.log()

  # CONV as a matmul: reshape -> matmul (sparse kernel) -> reshape
  def conv2d(self, in_c, out_c, kernel_size):
    kernel = tensor.randn(1,1,kernel_size, kernel_size) if isinstance(kernel_size, int) else tensor.randn(*kernel_size)
    tplz = kernel.sparse(*self.shape)
    out = self.reshape(-1,).dot(tplz)
    return out

  # combine tensors along axis: PAD to output shape -> ADD
  # TODO: work for multiple args
  def cat(self, *args, dim=0):  
    assert all(len(x.shape) == len(self.shape) for x in args)
    #extend self along dim, extend args along dim, then concatonate
    s = self.pad((self.shape[0],0), axis = dim)
    ot = args[0].pad((0,self.shape[0]), axis = dim)
    return s + ot

  def unsqueeze(self, axis): 
    dim = (self.shape[:axis] + (1,) + self.shape[axis:])
    return self.reshape(*dim)


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
    return topo

  def backward(self):
    DEBUG = os.getenv("DEBUG") 

    assert(self.grad.shape == (1,))
    self.grad = np.ones(self.grad.shape)
    for x in reversed(self.toposort()): 
      assert x.grad.shape == x.shape, \
        f"grad shape must match tensor shape in {x.grad.shape} != {x.shape} on {x.op.arg}"
      if DEBUG:
        in_s = list(n.shape for n in x.op.saved)
        print(f"op = <{x.op.arg}> in: {in_s} -> out: {x.data.shape} with grad: {x.grad.shape}")
      x.op.backward(x.grad, x.data)
      x.grad = np.zeros(x.grad.shape)

  def cast_op(self, fxn, x):
    x, y = self, x 
    y = y if isinstance(y, tensor) else tensor(y)
    x = x if isinstance(x, tensor) else tensor(x)
    if x.shape == y.shape: 
      return tensor(fxn.forward(x,y), op = fxn(saved = [x, y]))
    cst, shp, ot, axis = castable(x,y)
    # preserves casting order based on castable outputs
    if axis == 1: 
      cst = cst.cast(shp, ctx = shp)
    if axis == 0:
      ot = ot.cast(shp, ctx = shp)
    return tensor(fxn.forward(cst, ot), op = fxn(saved = [cst, ot]))

class Linear: 
  def __init__(self, in_shape, out_shape, bias=True):
    self.w = tensor.randn(in_shape, out_shape)
    self.b = tensor.randn(out_shape) if bias else None

  def __call__(self, x): 
    return x.dot(self.w) + self.b

# returns cast, target, and buffer
def castable(x, y): 
  assert is_castable(x.shape,y.shape), f"shapes {x.shape} and {y.shape} are not castable"
  out = np.broadcast_shapes(x.shape, y.shape)
  if x.shape != out:
    return x, out, y, 1
  if y.shape != out:
    return x, out, y, 0

def is_castable(x, y): 
  for a, b in zip(x[::-1], y[::-1]): 
    if a == 1 or b == 1 or a == b:
      pass
    else: 
      return False
  return True


