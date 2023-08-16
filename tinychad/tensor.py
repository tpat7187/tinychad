from __future__ import annotations
import numpy as np 
import os
import time
from typing import List, Optional, Tuple, Union, Type

DEBUG = os.getenv("DEBUG") 
LAZY = os.getenv("LAZY")

class OP: 
  def __init__(self, saved:Optional[Tuple[tensor, ...]]=None, ctx:Optional[int]=None):
    self.arg = type(self).__name__
    self.saved = np.array(saved)
    self.ctx = ctx

  def forward(x, y): return f"forward not implemented for {self.arg}" 
  def backward(self, out_grad, out): return f"backward not implemented for {self.arg}" 

  @classmethod
  def apply(self:Type[ops.OP], *x:tensor, lazy:Optional[bool] = False, **kwargs):
    if LAZY and lazy == False:
      lazyshape = ViewTracker.generate_view(self, *x, **kwargs)
      out = tensor(LazyBuffer(lazyshape, self), op = self(saved = [*x], ctx = list(kwargs.values())))
      return out
    if DEBUG: st = time.monotonic()
    out =  tensor(self.forward(*x, **kwargs), op = self(saved = [*x], ctx = list(kwargs.values())))
    if DEBUG: 
      et= time.monotonic() - st
      in_s = list(n.shape for n in out.op.saved)
      print("op = {:10} in: {:<45} out: {:<30} in: {:.2f}us".format(out.op.arg, str(in_s), str(out.data.shape), et*1e4))
    return out

import tinychad.ops as ops

# **** TENSOR CLASS ****
class tensor: 
  __slots__ = "data", "requires_grad", "op", "grad"
  def __init__(self, data: Union[np.ndarray, LazyBuffer, int, float, list], op:ops.OP = ops.LOAD(), requires_grad:Optional[bool] = False):
    if isinstance(data, (np.ndarray, LazyBuffer)): 
      self.data = data

    if isinstance(data, (int, list, float)): 
      self.data = np.array(data)

    self.grad, self.requires_grad, self.op = np.zeros(self.data.shape, dtype = np.float32), requires_grad, op

  @staticmethod
  def ones(*shape, **kwargs): return tensor(np.ones(*shape), **kwargs)

  @staticmethod
  def randn(*shape, **kwargs): return tensor(np.random.randn(*shape), **kwargs)

  @staticmethod
  def eye(shape, **kwargs): return tensor(np.eye(shape), **kwargs)

  @staticmethod
  def zeros(*shape, **kwargs): return tensor(np.zeros(*shape), **kwargs)

  @staticmethod
  def uniform(*shape,hi=1,lo=-1,**kwargs): return tensor(np.random.uniform(size=shape, low=lo, high=hi), **kwargs)

  @staticmethod
  def kaiming_uniform(*shape, a=0.01, **kwargs): 
    b = np.sqrt(3.0) * np.sqrt(2.0 / (1 + a**2)) / np.sqrt(np.prod(shape[1:]))
    return tensor.uniform(*shape, hi=b, lo=-b, **kwargs)

  @property
  def shape(self) -> tuple[int, ...]: return self.data.shape

  @property
  def dtype(self): return self.data.dtype

  @property
  def size(self): return self.data.size

  def __repr__(self): 
    return f"op = <{self.op.arg}>: shape = {self.shape}" #lazycache = {self._cache}"
  
  # TODO: getitem by tensor index
  def __getitem__(self, args): 
    if isinstance(args, int): return self.slice((args))
    elif isinstance(args, tuple): return self.slice(args)

  def __add__(self, x): return self.add(x)
  def __sub__(self,x): return self.sub(x)
  def __mul__(self,x): return self.mul(x)
  def __matmul__(self,x): return self.dot(x)
  def __truediv__(self,x): return self.div(x)
  def __neg__(self): return self.neg()

  def __radd__(self,x): return self.add(x, reverse=True)
  def __rsub__(self,x): return self.sub(x, reverse=True)
  def __rmul__(self,x): return self.mul(x, reverse=True)
  def __rtruediv__(self,x): return self.div(x, reverse=True)
  def __rmatmul__(self, x): return self.dot(x, reverse=True)

  def __iadd__(self,x): return self.add(x)
  def __isub__(self,x): return self.sub(x)
  def __imul__(self,x): return self.mul(x)

  # binary ops
  def add(self, x:Union[tensor, float], reverse=False) -> tensor: return self.cast_op(ops.ADD, x, reverse)
  def sub(self, x:Union[tensor, float], reverse=False) -> tensor: return self.cast_op(ops.SUB, x, reverse)
  def mul(self, x:Union[tensor, float], reverse=False) -> tensor: return self.cast_op(ops.MUL, x, reverse)
  def div(self, x:Union[tensor, float], reverse=False) -> tensor: return self.cast_op(ops.DIV, x, reverse)
  def dot(self, x:Union[tensor, float], reverse=False) -> tensor: return ops.MATMUL.apply(self, x) if reverse == False else ops.MATMUL.apply(x, self)

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

  def reshape_op(self, fxn:ops.OP, *args:tensor, lazy:Optional[bool] = False, **kwargs) -> tensor: 
    return fxn.apply(self, **kwargs, lazy=lazy)

  def T(self) -> tensor: return tensor(self.data.transpose())
  def matmul(self, x): return self.dot(x)
  def sigmoid(self) -> tensor: return self.exp().div(self.exp()+1)
  def square(self) -> tensor: return self*self

  def mean(self, axis:Optional[Union[Tuple[int, ...], int]]=None, keepdim:Optional[bool]=False) -> tensor:
    out = self.sum(axis=axis, keepdim=keepdim)
    ss = out * (np.prod(out.shape) / np.prod(self.shape))
    return ss

  def var(self, axis:Optional[Union[Tuple[int, ...], int]]=None, keepdim:Optional[bool]=False) -> tensor:
    mn = self.mean(axis=axis, keepdim=keepdim)
    ss = (self.sub(mn).square()).sum(axis=axis, keepdim=keepdim)
    out = ss / (np.prod(self.shape)/np.prod(ss.shape))
    return out

  def reciprocal(self) -> tensor: return 1 / self

  def _softmax(self, axis:int) -> Tuple[tensor, tensor, tensor]:
    m = self - self.max(axis=axis, keepdim=True)
    e = m.exp() 
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis:Optional[int]=-1) -> tensor:
    _, e, ss = self._softmax(axis) 
    return e.div(ss)
 
  def logsoftmax(self, axis:Optional[int]=-1) -> tensor:
    m, _, ss = self._softmax(axis) 
    return m - ss.log()

  # CONV as a matmul: input -> im2col MATMUL kernel.reshape(-1,1)
  # self <- input image, weight <- kernel, bias < - bias, return conv operation
  def conv2d(self, weight:tensor, bias:Optional[tensor]=None, padding=0, stride=1) -> tensor:
    N, Cin, H, W = self.shape
    Cout, _, k_h, k_w = weight.shape
    out_h, out_w = ((H + 2 * padding - k_h)//stride + 1), ((W + 2 * padding - k_w)//stride + 1)
    k, i, j = self.get_im2col_indices(k_h, k_w, padding=padding, stride=stride)
    x_padded = self.pad(((0,0), (0,0), (padding, padding), (padding,padding)))
    cols = x_padded[:, k, i, j].transpose(1,2,0).reshape(k_h * k_w * Cin, -1)
    out = (weight.reshape(Cout,-1).dot(cols)).reshape(Cout, out_h, out_w, N).transpose(3,0,1,2)
    if bias is not None: out = out + bias
    return out 

  def max_pool2d(self, kernel_size:Union[Tuple[int,...], int], stride:int=1) -> tensor:
    kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    stride = stride if stride != None else kernel_size[0] 
    N, Cin, H, W = self.shape
    k_h, k_w = kernel_size
    res = self.reshape(N, Cin, H // k_h, k_h, W // k_w, k_w)
    out = res.max(axis=3).max(axis=4)
    return out

  def avg_pool2d(self, kernel_size:Union[Tuple[int, ...], int], stride:Optional[int]=None) -> tensor:
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
  def get_im2col_indices(self, f_h:int, f_w:int, padding:Optional[int]=0, stride:Optional[int]=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
  
  # running mean/std are tensors of size (1,1,C,1)
  def batchnorm2d(self, weight:tensor, bias:Optional[tensor], running_mean:tensor, running_var:tensor, eps=1e-5, momentum=0) -> tensor:
    assert len(self.shape) == 4
    N,C,H,W = self.shape
    mean = self.mean(axis=(0,2,3),keepdim=True)
    var = (self - mean).square().mean(axis=(0,2,3),keepdim=True)
    x_h = (self - mean) / (var + eps)
    out = weight * x_h + bias if bias else weight*x_h
    return out

  def pad_to(self, shape:Tuple[int, ...]) -> tensor:
    in_s, o_s, ss = self.shape, shape, 0
    p_w = [[0,0] for _ in range(len(self.shape))]
    for i,j in zip(self.shape, shape):
      p_w[ss][1] = j-i
      ss+=1 # cringe code please fix
    return self.pad(p_w)

  def cat(self, *args:tensor, axis:Optional[int]=0) -> tensor:
    assert all(len(x.shape) == len(self.shape) for x in args)
    cache, out = [self, *args], []
    for j in range(len([self, *args])):
      pad_t = [[0,0] for _ in range(len(self.shape))]
      pad_t[axis] = [self.shape[0]*j, self.shape[0]*(len(args)*1-j)]
      out.append(cache[j].pad(pad_t))
    return sum(out)

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

  def unsqueeze(self, axis: int) -> tensor:
    dim = (self.shape[:axis] + (1,) + self.shape[axis:])
    return self.reshape(*dim)

  def squeeze(self, axis: int) -> tensor:
    assert self.shape[axis] == 1, f"cannot squeeze {self} along axis with value {self.shape[axis]}"
    dim = (self.shape[:axis] + self.shape[1+axis:])
    return self.reshape(*dim)

  def flatten(self): return self.reshape(-1,)
 
  def toposort(self) -> Tuple[tensor, ...]: 
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
    # potential refactor: ops.backward act directly on the saved tensors and call IADD each time
    # if we can remove the need to allocate the memory for all grads on creation it would increase training time
    # ops.backward return ndarrays, if grad != None iadd else assign
    # after grads are passed backward we simply set current grad back to None
    self.exec()
    assert(self.grad.shape == (1,))
    self.grad = np.ones(self.grad.shape)
    for x in reversed(self.toposort()): 
      assert x.grad.shape == x.shape, \
        f"grad shape must match tensor shape in {x.grad.shape} != {x.shape} on {x.op.arg}"
      x.op.backward(x.grad, x.data)
      x.grad = np.zeros(x.grad.shape) if x.requires_grad == False else x.grad

  def cast_op(self, fxn: ops.op, x: tensor, reverse:bool) -> tensor:
    x, y = (self, x) if reverse == False else (x, self)
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
  
  # takes labels and logprobs (doesnt need one hot encoding)
  def NLLLoss(self, y_true: tensor) -> tensor:
    batch_s = self.shape[0]
    idx_probs = self[np.arange(batch_s).astype(np.int32), y_true.data]
    loss = (batch_s / -idx_probs.sum()).reciprocal()
    return loss

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

  def get_buffers(self) -> Tuple: 
    assert LAZY, "cannot get buffers without lazy evaluation enabled"
    cache = set()
    def _get_buffers(s):
      if id(s) not in cache and type(s.op) != ops.LOAD:
        cache.add((hex(id(s)), s.shape, type(s.op)))
        [_get_buffers(child) for child in s.op.saved if type(s.op) != ops.LOAD]
    _get_buffers(self)
    return cache

  def exec(self) -> tensor:
    if self.realized(): return self
    BinaryOPS = [ops.ADD, ops.SUB, ops.MUL, ops.DIV, ops.MATMUL]
    UnaryOPS = [ops.RELU, ops.LOG, ops.EXP, ops.NEG, ops.SQRT]
    ShapeOPS = [ops.SUM, ops.MAX]
    ReshapeOPS = [ops.RESHAPE, ops.SLICE, ops.TRANSPOSE, ops.PAD, ops.CAST]

    for f in self.op.saved: 
      if not f.realized(): f.exec()
    if type(self.op) in (BinaryOPS + UnaryOPS):
      s = self.op.apply(*[j for j in self.op.saved], lazy = True)
    elif type(self.op) in ShapeOPS:
      axis, keepdim = self.op.ctx[0], self.op.ctx[1]
      s = self.op.apply(*[j for j in self.op.saved], axis=axis, keepdim=keepdim, lazy = True)
    elif type(self.op) in ReshapeOPS: 
      s = self.op.saved[0].reshape_op(self.op, args = self.op.ctx[0], lazy = True)
    self.data = s.data
    return self
  
  # if NOT lazybuffer then its realized
  def realized(self) -> bool: return not isinstance(self.data, LazyBuffer)

# for NN layers the optimizer will set requires_grad to True from statedict
class Linear: 
  def __init__(self, in_shape, out_shape, bias=True):
    bound = 1 / np.sqrt(out_shape)
    self.w = tensor.uniform(in_shape, out_shape, lo=-bound, hi=bound)
    self.b = tensor.uniform(out_shape, lo=-bound, hi=bound) if bias else None

  def __call__(self, x: tensor) -> tensor: 
    return x.dot(self.w).add(self.b)

class Conv2d: 
  def __init__(self, in_channels, out_channels, kernel_size:Union[Tuple[int, int], int], padding=1, stride=1, bias=True):
    self.padding, self.stride = padding, stride
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    bound = 1 / (in_channels * np.prod(kernel_size))
    self.w = tensor.kaiming_uniform(out_channels, in_channels, *kernel_size, a = np.sqrt(5))
    self.b = tensor.uniform(1,out_channels,1,1, hi=bound, lo=-bound) if bias else None 

  def __call__(self, x:tensor) -> tensor:
    return x.conv2d(weight=self.w, bias=self.b, padding=self.padding, stride=self.stride)

class BatchNorm2d:
  def __init__(self, num_features, affine=True, momentum=0, track=False, eps=1e-5):
    self.momentum, self.eps, self.track = momentum, eps, track
    if affine:
      self.w = tensor.ones((1,num_features,1,1))
      self.b = tensor.ones((1,num_features,1,1))
    else:
      self.w, self.b = None, None
    self.running_mean = tensor.zeros(num_features)
    self.running_var = tensor.zeros(num_features)

  def __call__(self, x:tensor) -> tensor:
    if self.track:
      self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum #.mul(currentmean)
      self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum #.mul(currentvar)

    return x.batchnorm2d(self.w, self.b, self.running_mean, self.running_var, self.eps, self.momentum)

# returns cast, target, and buffer
def castable(x: tensor, y: tensor) -> Tuple[tensor, Tuple[int, ...], tensor, int]:
  assert is_castable(x.shape,y.shape), f"shapes {x.shape} and {y.shape} are not castable"
  out = np.broadcast_shapes(x.shape, y.shape)
  if x.shape != out: return x, out, y, 1
  if y.shape != out: return x, out, y, 0

def is_castable(x: Tuple[int, ...], y:Tuple[int, ...]) -> bool:
  for a, b in zip(x[::-1], y[::-1]): 
    if a == 1 or b == 1 or a == b:
      pass
    else: 
      return False
  return True

class LazyBuffer: 
  def __init__(self, shape:Tuple[int, ...], op:ops.OP):
    self.shape, self.op = shape, op

class ViewTracker: 
  @classmethod
  def generate_view(self, op: ops.OP, *args, **kwargs) -> Tuple[int, ...]:
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
    
  def _reshape(in_s: tensor, kwargs: dict) -> Tuple[int, ...]:
    arg, in_s = list(kwargs['args']), list(in_s.shape)
    out_s = tuple(arg)
    if -1 in arg:
      idx = arg.index(-1)
      _cur = np.prod([j for j in arg if j != -1])
      arg[idx] = np.prod(in_s)//_cur
      out_s = tuple(arg)
    return out_s

  def _slice(in_s: tensor, kwargs: dict) -> Tuple[int, ...]:
    arg = kwargs['args'][0] if not isinstance(kwargs['args'][0], int) else kwargs['args'][0]
    # TEMPORARY HACK 
    # we shouldnt be executing the slice to have it done, we need to interate through each of the slices and then calculate the output shape
    # numpy has broadcasting rules for how slices can be reduced EG: (1,1,5,5) -> (1,9,9) im2col the (9,1) 2nd index and the (9,9)(9,9) 3rd and 4th get broadcasted
    out_s = np.empty(in_s.shape)[arg].shape
    out_s = (1,) if out_s == () else out_s
    return out_s 

  def _transpose(in_s: tensor, kwargs: dict) -> Tuple[int, ...]:
    arg, in_s = list(kwargs['args']), list(in_s.shape)
    return tuple([in_s[i] for i in arg])

  def _pad(in_s: tensor, kwargs: dict) -> Tuple[int, ...]:
    return tuple([i+j for i, j in zip([sum(list(j)) for j in list(kwargs['args'])], (list(in_s.shape)))])
    
  def _cast(in_s: tensor, kwargs: dict) -> Tuple[int, ...]:
    return tuple(kwargs['args'])
