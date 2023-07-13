import numpy as np 
import os
import time

class OP: 
  def __init__(self, saved = None, ctx = None):
    self.arg = type(self).__name__
    self.saved = np.array(saved)
    self.ctx = ctx

  def forward(x, y): return f"forward not implemented for {self.arg}" 
  def backward(self, out_grad, out): return f"backward not implemented for {self.arg}" 

  @staticmethod
  def apply(fxn, x, *args, **kwargs): 
    return tensor(fxn.forward(x, *args), op = fxn(saved = [x, *args], ctx = kwargs))

import tinychad.ops as ops

#### TENSOR CLASS ####
class tensor: 
  def __init__(self, data, op = ops.LOAD(), requires_grad = False):
    self.data, self.op = np.array(data, dtype = np.float32), op
    self.grad, self.requires_grad = np.zeros(self.data.shape, dtype = np.float32), requires_grad

  def ones(*shape, **kwargs): return tensor(np.ones(*shape), **kwargs)
  def randn(*shape, **kwargs): return tensor(np.random.randn(*shape), **kwargs)
  def eye(shape, **kwargs): return tensor(np.eye(shape), **kwargs)
  def zeros(*shape, **kwargs): return tensor(np.zeros(*shape), **kwargs)

  def to_lazy(self): return LazyTensor(self)

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

  def dot(self, x): return OP.apply(ops.MATMUL, self, x)
  def matmul(self, x): return self.dot(x)

  # unary ops
  def relu(self): return OP.apply(ops.RELU, self)
  def exp(self):  return OP.apply(ops.EXP, self)
  def log(self):  return OP.apply(ops.LOG, self)
  def neg(self):  return OP.apply(ops.NEG, self)

  # shape ops (changes shape and content)
  def max(self, axis = None, keepdim = False): return tensor(ops.MAX.forward(self, axis, keepdim), op = ops.MAX(saved = [self,], ctx=[axis, keepdim]))
  def sum(self, axis = None, keepdim = False): return tensor(ops.SUM.forward(self, axis, keepdim), op = ops.SUM(saved = [self,], ctx=[axis, keepdim]))

  # reshape ops (changes shape, content does not change, sparse -> circular matrix for conv)
  def reshape(self, *shape) : return tensor(ops.RESHAPE.forward(self, *shape), op = ops.RESHAPE(saved = [self,]))
  def slice(self, *args) : return tensor(ops.SLICE.forward(self, *args), op = ops.SLICE(saved = [self,], ctx = args))
  def pad(self, args): return tensor(ops.PAD.forward(self, args), op = ops.PAD(saved = [self,], ctx = args))
  def roll(self, shift, axis): return tensor(ops.ROLL.forward(self, shift, axis), op = ops.ROLL(saved = [self,], ctx = [shift, axis]))
  def transpose(self, *order): return tensor(ops.TRANSPOSE.forward(self, order), op = ops.TRANSPOSE(saved = [self,], ctx = order))

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

  # CONV as a matmul: input -> im2col MATMUL kernel.reshape(-1,1)
  # self <- input image, weight <- kernel, bias < - bias, return conv operation
  def conv2d(self, weight, padding=1, stride=1):
    N, Cin, H, W = self.shape
    Cout, _, k_h, k_w = weight.shape
    out_h, out_w = (H + 2 * padding - k_h + 1), (W + 2 * padding - k_w + 1)
    k, i, j = self.get_im2col_indices(k_h, k_w, padding=padding, stride=stride)
    x_padded = self.pad(((0,0), (0,0), (padding, padding), (padding,padding)))
    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1,2,0).reshape(k_h * k_w * Cin, -1)
    out = (weight.reshape(Cout,-1).dot(cols)).reshape(Cout, out_h, out_w, N).transpose(3,0,1,2)
    return out

  # this doesnt work (it looks awfully similar to conv2d)
  def max_pool2d(self, kernel, padding=1, stride=1):
    N, Cin, H, W = self.shape
    Cout, _, k_h, k_w = kernel.shape
    out_h, out_w = (H + 2 * padding - k_h + 1), (W + 2 * padding - k_w + 1)
    k, i, j = self.get_im2col_indices(k_h, k_w, padding=padding, stride=stride)
    x_padded = self.pad(((0,0), (0,0), (padding, padding), (padding,padding)))
    cols = x_padded[:, k, i, j].transpose(1,2,0).reshape(k_h * k_w * Cin, -1)
    cols_max = np.argmax(cols.data, axis=0)
    out = cols[cols_max, range(cols_max.size)]
    out = out.reshape(out_h, out_w, N, Cout).transpose(2,3,0,1)
    return out

  # input image, kernel_h, kernel_w
  def get_im2col_indices(self, f_h, f_w, padding=1, stride=1):
    N, C, H, W = self.shape
    out_height = int((H+2 * padding - f_h) / stride + 1)
    out_width = int((W+2 * padding - f_w) / stride + 1)
    i0 = np.repeat(np.arange(f_h), f_w)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(f_w), f_h * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), f_h * f_w).reshape(-1,1)
    return (k, i, j)

  def pad_to(self, shape): 
    in_s, o_s, ss = self.shape, shape, 0
    p_w = [[0,0] for _ in range(len(self.shape))]
    for i,j in zip(self.shape, shape):
      p_w[ss][1] = j-i
      ss+=1 # cringe code please fix
    return self.pad(p_w)

  def cat(self, *args, axis=0):  
    assert all(len(x.shape) == len(self.shape) for x in args)
    out_shape, args = list(self.shape), list(args)
    out_shape[axis] = out_shape[axis]*(len(args)+1)
    out_t = [[0,0] for _ in range(len(self.shape))]
    out_t[axis][1] = args[0].shape[axis] * len(args)
    out = self.pad(out_t)
    for j in range(len(args)): 
      args[j] = args[j].pad(out_t).roll((j+1)*args[j].shape[axis], axis)
      out += args[j]
    return out


  # input padded kernel and input size, output doubly blocked circulant matrix
  def tpltz(self, input_size, kernel_size):
    blocks, tpltz =  [], []
    for j in  range(self.shape[0]):
      r = self[j,:].reshape(-1,1)
      rl = [] 
      for j in range(r.shape[0]-kernel_size[0]):
        rl.append(r.roll(j+1, axis=0))
      blocks.append(r.cat(*rl, axis=1))
    blocks = blocks[::-1]
    block = tensor.cat(*[_ for _ in blocks], axis=0)
    for _ in range(input_size[0]):
      tpltz.append(block.roll(_*self.shape[1], axis=0))
    out = tensor.cat(*[_ for _ in tpltz], axis=1)
    return out

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
      x.grad = np.zeros(x.grad.shape) if x.requires_grad == False else x.grad

  def _apply(self, fxn, x): return fxn.forward(self, x)

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

# no idea what im doing here
# here is the idea, wrap the tensor object, delay computation through gathering ops
# realize the output buffer by executing all ops one after the other
# should allow us to get the # buffers needed for LLVM backend without wasting compute
class LazyTensor:
  def __init__(self, tensor = None, cache = None):
    self.tensor = tensor
    self.cache = [] if cache is None else cache
  
  def lazy_op(self, fxn, x):
    out = LazyTensor() 
    out.cache.append([fxn, self, x])
    return out

  def register(self): 
    for x in self.cache: 
      out = x[1].tensor._apply(x[0], x[1].tensor)
    return out

  def add(self, x): 
    return self.lazy_op(ops.ADD, x)


# for NN layers the optimizer will set requires_grad to True from statedict
class Linear: 
  def __init__(self, in_shape, out_shape, bias=True):
    self.w = tensor.randn(in_shape, out_shape)
    self.b = tensor.randn(out_shape) if bias else None

  def __call__(self, x): 
    return x.dot(self.w).add(self.b)

class Conv2d: 
  def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias = True):
    self.padding, self.stride = padding, stride
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    self.w = tensor.randn(1, out_channels, *kernel_size)
    self.b = tensor.randn(out_channels) if bias else None 

  def __call__(self, x): 
    return x.conv2d(weight=self.w, padding=self.padding, stride=self.stride).add(self.b)

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

