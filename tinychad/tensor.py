import numpy as np 
import os
import time
from typing import NewType, Type 

DEBUG = os.getenv("DEBUG") 
LAZY = os.getenv("LAZY")

class OP: 
  def __init__(self, saved = None, ctx = None):
    self.arg = type(self).__name__
    self.saved = np.array(saved)
    self.ctx = ctx

  def forward(x, y): return f"forward not implemented for {self.arg}" 
  def backward(self, out_grad, out): return f"backward not implemented for {self.arg}" 

  @classmethod
  def apply(self, *x, lazy = False, **kwargs):
    if LAZY and lazy == False:
      lazyshape = ViewTracker.generate_view(self, *x, **kwargs)
      # TODO: find a way to do this without allocating memory, we're delaying computation but keeping memoryalloca
      out = tensor(np.empty(lazyshape), op = self(saved = [*x], ctx = list(kwargs.values())))
      out._cache, out._lazyshape = x, lazyshape
      return out
    if DEBUG: st = time.monotonic()
    out =  tensor(self.forward(*x, **kwargs), op = self(saved = [*x], ctx = list(kwargs.values())))
    if DEBUG: 
      et= time.monotonic() - st
      in_s = list(n.shape for n in out.op.saved)
      print("op = {:10} in: {:<45} out: {:<30} in: {:.2f}us".format(out.op.arg, str(in_s), str(out.data.shape), et*1e4))
    return out


import tinychad.ops as ops

class ViewTracker: 
  @classmethod
  def generate_view(self, op, *args, **kwargs):
    # we should use enums before i go insane
    BinaryOPS = [ops.ADD, ops.SUB, ops.MUL, ops.DIV, ops.MATMUL]
    UnaryOPS = [ops.RELU, ops.LOG, ops.EXP, ops.NEG, ops.SQRT]
    ShapeOPS = [ops.SUM, ops.MAX]
    ReshapeOPS = [ops.RESHAPE, ops.SLICE, ops.TRANSPOSE, ops.PAD, ops.CAST]

    # is this cringe
    ReshapeOPHelpers = {
      ops.RESHAPE: self._reshape,
      ops.SLICE: self._slice,
      ops.TRANSPOSE: self._transpose,
      ops.PAD: self._pad,
      ops.CAST: self._cast,
    }

    args = list(args)
    if op in BinaryOPS:
      assert args[0].shape[1] == args[1].shape[0] if op == ops.MATMUL else args[0].shape == args[1].shape
      out_s = (args[0].shape[0], args[1].shape[1]) if op == ops.MATMUL else args[0].shape 
      return out_s
    elif op in UnaryOPS: 
      out_s = args[0].shape
      return out_s
    elif op in ShapeOPS:
      axis, keepdim = kwargs['axis'], kwargs['keepdim']
      if axis is None: out_s = (1,)
      else:
        nx = list(axis) if isinstance(axis, tuple) else [axis]
        l = list(args[0].shape)
        for j in nx: l[j] =0 
        out_s = tuple([i for i in l if i!=0]) if keepdim == False else tuple([1 if i == 0 else i for i in l])
      return out_s
    elif op in ReshapeOPS: 
      return ReshapeOPHelpers[op](args[0], kwargs)
    
  def _reshape(in_s, kwargs): 
    arg, in_s = list(kwargs['args']), list(in_s.shape)
    out_s = tuple(arg)
    if -1 in arg:
      idx = arg.index(-1)
      _cur = np.prod([j for j in arg if j != -1])
      arg[idx] = np.prod(in_s)//_cur
      out_s = tuple(arg)
    return out_s

  def _slice(in_s, kwargs): 
    arg = kwargs['args'][0] if not isinstance(kwargs['args'][0], int) else kwargs['args'][0]
    # TEMPORARY HACK 
    # we shouldnt be executing the slice to have it done, we need to interate through each of the slices and then calculate the output shape
    # numpy has broadcasting rules for how slices can be reduced EG: (1,1,5,5) -> (1,9,9) im2col the (9,1) 2nd index and the (9,9)(9,9) 3rd and 4th get broadcasted
    out_s = np.empty(in_s.shape)[arg].shape
    out_s = (1,) if out_s == () else out_s
    return out_s 

  def _transpose(in_s, kwargs):
    arg, in_s = list(kwargs['args']), list(in_s.shape)
    return tuple([in_s[i] for i in arg])

  def _pad(in_s, kwargs):
    return tuple([i+j for i, j in zip([sum(list(j)) for j in list(kwargs['args'])], (list(in_s.shape)))])
    
  def _cast(in_s, kwargs):
    return tuple(kwargs['args'])


# **** TENSOR CLASS ****
class tensor: 
  def __init__(self, data, op = ops.LOAD(), requires_grad = False):
    self.data = np.array(data, dtype=np.float32)
    self.grad, self.requires_grad, self.op = np.zeros(self.data.shape, dtype = np.float32), requires_grad, op

    if LAZY: 
      self._cache, self._lazyshape = [], self.data.shape if type(op) == ops.LOAD else ()

  def ones(*shape, **kwargs): return tensor(np.ones(*shape), **kwargs)
  def randn(*shape, **kwargs): return tensor(np.random.randn(*shape), **kwargs)
  def eye(shape, **kwargs): return tensor(np.eye(shape), **kwargs)
  def zeros(*shape, **kwargs): return tensor(np.zeros(*shape), **kwargs)
  def uniform(*shape,hi=1,lo=-1,**kwargs): return tensor(np.random.uniform(size=shape, low=lo, high=hi), **kwargs)

  def kaiming_uniform(*shape, a=0.01, **kwargs): 
    b = np.sqrt(3.0) * np.sqrt(2.0 / (1 + a**2)) / np.sqrt(np.prod(shape[1:]))
    return tensor.uniform(*shape, hi=b, lo=-b, **kwargs)

  @property
  def shape(self): return self.data.shape if not LAZY else self._lazyshape

  @property
  def dtype(self): return self.data.dtype

  @property
  def size(self): return self.data.size

  def __repr__(self): 
    return f"op = <{self.op.arg}>: shape = {self.shape}" #lazycache = {self._cache}"

  def __getitem__(self, args): 
    if isinstance(args, int): return self.slice((args))
    if isinstance(args, tuple): return self.slice(args)

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
  def dot(self, x): return ops.MATMUL.apply(self, x)

  # unary ops
  def relu(self): return ops.RELU.apply(self)
  def sqrt(self):  return ops.SQRT.apply(self)
  def exp(self):  return ops.EXP.apply(self)
  def log(self):  return ops.LOG.apply(self)
  def neg(self):  return ops.NEG.apply(self)

  # shape ops (changes shape and content)
  def max(self, axis=None, keepdim=False): return ops.MAX.apply(self, axis=axis, keepdim=keepdim)
  def sum(self, axis=None, keepdim=False): return ops.SUM.apply(self, axis=axis, keepdim=keepdim)

  # reshape ops (changes shape, content does not change, sparse -> circular matrix for conv)
  def reshape(self, *args) : return self.reshape_op(ops.RESHAPE, args = args)
  def slice(self, *args) : return self.reshape_op(ops.SLICE, args = args)
  def pad(self, args) : return self.reshape_op(ops.PAD, args = args)
  def transpose(self, *args) : return self.reshape_op(ops.TRANSPOSE, args = args)
  def cast(self, args) : return self.reshape_op(ops.CAST, args = args)

  def reshape_op(self, fxn, *args, lazy = False, **kwargs): 
    return fxn.apply(self, **kwargs, lazy=lazy)

  def T(self): return tensor(self.data.transpose())
  def argmax(self, axis = None): return self.data.argmax(axis=axis)
  def matmul(self, x): return self.dot(x)
  def sigmoid(self): return self.exp().div(self.exp()+1)
  def square(self): return self*self

  def mean(self, axis=None, keepdim=False): 
    out = self.sum(axis=axis, keepdim=keepdim)
    ss = out * (np.prod(out.shape) / np.prod(self.shape))
    return ss

  def var(self, axis=None, keepdim=False): 
    mn = self.mean(axis=axis, keepdim=keepdim)
    ss = (self.sub(mn).square()).sum(axis=axis, keepdim=keepdim)
    out = ss / (np.prod(self.shape)/np.prod(ss.shape))
    return out

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
  def conv2d(self, weight, bias=None, padding=0, stride=1):
    N, Cin, H, W = self.shape
    Cout, _, k_h, k_w = weight.shape
    out_h, out_w = ((H + 2 * padding - k_h)//stride + 1), ((W + 2 * padding - k_w)//stride + 1)
    k, i, j = self.get_im2col_indices(k_h, k_w, padding=padding, stride=stride)
    x_padded = self.pad(((0,0), (0,0), (padding, padding), (padding,padding)))
    cols = x_padded[:, k, i, j].transpose(1,2,0).reshape(k_h * k_w * Cin, -1)
    out = (weight.reshape(Cout,-1).dot(cols)).reshape(Cout, out_h, out_w, N).transpose(3,0,1,2)
    if bias is not None: 
      out = out + bias
    return out 

  # TODO: this still doesnt work for N > 1
  def max_pool2d(self, kernel_size, stride=1):
    kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    stride = stride if stride != None else kernel_size[0] 
    N, Cin, H, W = self.shape
    k_h, k_w = kernel_size
    res = self.reshape(N, Cin, H // k_h, k_h, W // k_w, k_w)
    out = res.max(axis=3).max(axis=4)
    return out

  def avg_pool2d(self, kernel_size, stride=None):
    kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    stride = stride if stride != None else kernel_size[0] 
    N, Cin, H, W = self.shape
    k_h, k_w = kernel_size
    out_h, out_w = ((H - k_h)//stride) + 1, ((W - k_w)//stride) + 1
    self = self.reshape(N*Cin, 1, H, W)
    k, i, j = self.get_im2col_indices(k_h, k_w, stride=stride)
    cols = self[:, k, i, j].transpose(1,2,0).reshape(k_h * k_w, Cin, -1)
    out = cols.mean(axis=0).reshape(out_h, out_w, N, Cin).transpose(2,3,0,1)
    return out

  # input image, kernel_h, kernel_w
  def get_im2col_indices(self, f_h, f_w, padding=0, stride=1):
    N, C, H, W = self.shape
    out_height = ((H+2 * padding - f_h) //stride) + 1
    out_width = ((W+2 * padding - f_w) //stride) + 1
    i0 = np.repeat(np.arange(f_h), f_w)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(f_w), f_h * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), f_h * f_w).reshape(-1,1)
    return (k, i, j)

  def batchnorm2d(self, weight, bias = None, eps=1e-5):
    assert len(self.shape) == 4
    mn = self.mean(axis=(0,2,3), keepdim=True)
    vr = self.var(axis=(0,2,3), keepdim=True)
    norm = (self - mn) / (vr + eps).sqrt()
    out = weight * norm + bias if bias is not None else weight * norm
    return out

  def pad_to(self, shape): 
    in_s, o_s, ss = self.shape, shape, 0
    p_w = [[0,0] for _ in range(len(self.shape))]
    for i,j in zip(self.shape, shape):
      p_w[ss][1] = j-i
      ss+=1 # cringe code please fix
    return self.pad(p_w)

  # TODO: fix this with transpose instead of roll
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

  def flatten(self): return self.reshape(-1,)
 
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
    assert(self.grad.shape == (1,))
    self.grad = np.ones(self.grad.shape)
    for x in reversed(self.toposort()): 
      assert x.grad.shape == x.shape, \
        f"grad shape must match tensor shape in {x.grad.shape} != {x.shape} on {x.op.arg}"
      x.op.backward(x.grad, x.data)
      x.grad = np.zeros(x.grad.shape) if x.requires_grad == False else x.grad

  def cast_op(self, fxn, x):
    x, y = self, x 
    y = y if isinstance(y, tensor) else tensor([y])
    x = x if isinstance(x, tensor) else tensor([x])
    if x.shape == y.shape: return fxn.apply(x,y)
    cst, shp, ot, axis = castable(x,y)
    # preserves casting order based on castable outputs
    if axis == 1: cst = cst.cast(shp)
    if axis == 0: ot = ot.cast(shp)
    return fxn.apply(cst, ot)
  
  # one hot encodes tensor acts on data so as to not join computation graph
  @classmethod
  def OHE(self, real, classes):
    r = real.flatten().astype(np.int32)
    y = np.zeros((r.shape[0], classes), np.float32)
    y[range(y.shape[0]), r] = -1.0*classes
    y = y.reshape(list(real.shape) + [classes])
    y = tensor(y)
    return y

  def NLLLoss(self, real):
    classes = self.shape[-1]
    r = real.flatten().astype(np.int32)
    y = np.zeros((r.shape[0], classes), np.float32)
    y[range(y.shape[0]), r] = -1.0*classes
    y = y.reshape(list(real.shape) + [classes])
    y = tensor(y)
    return self.mul(y).mean()

  def cross_entropy(self, real, reduction = 'mean', smoothing = 0.0):
    classes = self.shape[-1]
    r = real.flatten().astype(np.int32)
    y = np.zeros((r.shape[0], classes), np.float32)
    y[range(y.shape[0]), r] = -1.0*classes
    y = y.reshape(list(real.shape) + [classes])
    y = tensor(y)

    if reduction=='none': 
      return -self.logsoftmax(axis=1).mul(y).sum(axis=1)
    if reduction=='sum': 
      return -self.logsoftmax(axis=1).mul(y).sum(axis=1).sum()
    return self.logsoftmax(axis=1).mul(y).sum(axis=1,keepdim=True).mean()

  def get_buffers(self): 
    assert LAZY, "cannot get buffers without lazy evaluation enabled"
    cache = set()
    def _get_buffers(s):
      if id(s) not in cache and type(s.op) != ops.LOAD:
        cache.add((hex(id(s)), s.shape))
        [_get_buffers(child) for child in s._cache if type(s.op) != ops.LOAD]
    _get_buffers(self)
    return cache

  def exec(self):
    assert LAZY
    BinaryOPS = [ops.ADD, ops.SUB, ops.MUL, ops.DIV, ops.MATMUL]
    UnaryOPS = [ops.RELU, ops.LOG, ops.EXP, ops.NEG, ops.SQRT]
    ShapeOPS = [ops.SUM, ops.MAX]
    ReshapeOPS = [ops.RESHAPE, ops.SLICE, ops.TRANSPOSE, ops.PAD, ops.CAST]

    for f in self._cache: 
      if not f.realized(): f.exec()
    if type(self.op) in (BinaryOPS + UnaryOPS):
      s = self.op.apply(*[j for j in self._cache], lazy = True)
    elif type(self.op) in ShapeOPS:
      axis, keepdim = self.op.ctx[0], self.op.ctx[1]
      s = self.op.apply(*[j for j in self._cache], axis=axis, keepdim=keepdim, lazy = True)
    elif type(self.op) in ReshapeOPS: 
      s = self._cache[0].reshape_op(self.op, args = self.op.ctx[0], lazy = True)

    self.data = s.data
    self._cache = []
    return self
  
  # TODO: when writing lazybuffer class, a lazy tensor is realized if both items in the cache are not lazybuffers
  def realized(self): return len(self._cache) == 0

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
    self.w = tensor.randn(out_channels, in_channels, *kernel_size)
    self.w = tensor.kaiming_uniform(out_channels, in_channels, *kernel_size, a = np.sqrt(5))
    self.b = tensor.randn(1,out_channels,1,1) if bias else None 

  def __call__(self, x):
    return x.conv2d(weight=self.w, bias=self.b, padding=self.padding, stride=self.stride)

class BatchNorm2d:
  def __init__(self, num_features, affine=True):
    if affine == True:
      self.w = tensor.ones((1,num_features,1,1))
      self.b = tensor.ones((1,num_features,1,1))
    else:
      self.w, self.b = None, None

  # TODO: add self.training property st we can track running mean/std
  def __call__(self, x):
    return x.batchnorm2d(weight=self.w, bias=self.b)

# returns cast, target, and buffer
def castable(x, y): 
  assert is_castable(x.shape,y.shape), f"shapes {x.shape} and {y.shape} are not castable"
  out = np.broadcast_shapes(x.shape, y.shape)
  if x.shape != out: return x, out, y, 1
  if y.shape != out: return x, out, y, 0

def is_castable(x, y): 
  for a, b in zip(x[::-1], y[::-1]): 
    if a == 1 or b == 1 or a == b:
      pass
    else: 
      return False
  return True

