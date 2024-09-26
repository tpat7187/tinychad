from __future__ import annotations
import os, math, ctypes, subprocess, tempfile
import numpy as np 
from typing import Union, Tuple, Optional, List, Dict, Any
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS, LoadOPS, Ops
from tinychad.tokenizer import Tokenizer
from tinychad.codegen import ExecuteCProgram, C_Codegen

class LoadOP: 
  __slots__ = "shape", "arg", "loadop"
  def __init__(self, shape, loadop, arg=None): 
    self.shape, self.arg, self.loadop = shape, arg, loadop 

  def __repr__(self): return str(self.loadop)

  @classmethod
  def alloc_raw(self, shape:Tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)

  @classmethod
  def alloc_const(self, shape:Tuple[int, ...], arg:int) -> np.ndarray:
    return np.full(shape, arg).astype(np.float32)

  @classmethod
  def alloc_rand(self, shape:Tuple[int, ...], arg:int) -> np.ndarray:
    return np.random.randn(*shape).astype(np.float32)

LoadOPSAllocator = {
  LoadOPS.RAND: LoadOP.alloc_rand,
  LoadOPS.CONST: LoadOP.alloc_const
}

OPT = os.getenv("OPT", 0)

# optype : node that has no buffer that performs an operation

class Function: 
  def __init__(self, optype, srcs): 
    self.optype = optype # general type EG: BinaryOPS
    self.srcs: Union[Function, Buffer] = srcs

  def __repr__(self):
    return f"<{type(self).__name__}: srcs = <{self.srcs}>]>"


MERGE_ELEMENTWISE_OPS = 1

# this will set the children depending on something
def create_buffer(shape, optype, func, ctx:Optional[List[Any]]=None): 
  return Buffer(shape, optype=optype, fxn = func, ctx=ctx)

# TODO: remove reshapes
class Buffer: 
  __slots__ = "shape", "optype", "children", "data", "ctx", "strides", "fxn"
  def __init__(self, shape, optype, children:Optional[List[Buffer]]=None, data:Optional[np.ndarray]=None, ctx=None, fxn=None): 
      self.shape, self.optype, self.children, self.ctx, self.data, = shape, optype, children, ctx, data
      self.strides = ViewTracker.generate_strides(shape)

      self.fxn = fxn

      # children is the link between kernels
      # loads can always be merged into Function

  @property 
  def dtype(self): return np.float32

  @property
  def size(self): return math.prod(self.shape)

  def __repr__(self): 
    return f"<{type(self).__name__}: op = <{self.optype}>: [shape = {self.shape}, strides = {self.strides}]>"
  
  def __add__(self, x:Buffer) -> Buffer: return self.binary_op(BinaryOPS.ADD, x)
  def __radd__(self, x:Buffer) -> Buffer: return x.binary_op(BinaryOPS.ADD, self)
  def __sub__(self, x:Buffer) -> Buffer: return self.binary_op(BinaryOPS.SUB, x)
  def __rsub__(self, x:Buffer) -> Buffer: return x.binary_op(BinaryOPS.SUB, self)
  def __mul__(self, x:Buffer) -> Buffer: return self.binary_op(BinaryOPS.MUL, x)
  def __rmul__(self, x:Buffer) -> Buffer: return x.binary_op(BinaryOPS.MUL, self)
  def __truediv__(self, x:Buffer) -> Buffer: return self.binary_op(BinaryOPS.DIV, x)
  def __rtruediv__(self, x:Buffer) -> Buffer: return x.binary_op(BinaryOPS.DIV, self)
  def __neg__(self) -> Buffer: return self.unary_op(UnaryOPS.NEG)

  # fused operations do not return a buffer, we simply edit our current buffer
  # the only binop that changes shape is a matmul, ill implement that later 
  # function is created every call

  def binary_op(self, optype, x:Buffer) -> Buffer: 
    src: Tuple[Buffer] = [self, x]

    # TODO: need a function that creates buffers and assigns children
    if MERGE_ELEMENTWISE_OPS and (x.children is None and self.optype in BinaryOPS): 
      src = [self.fxn, x]

    return create_buffer(self.shape, optype, Function(optype, src))

  def unary_op(self, optype) -> Buffer: return create_buffer(self.shape, optype, Function(optype, self))

  def shape_op(self, optype, axis, keepdim) -> Buffer: 
    if axis is not None and axis < 0: axis = np.arange(len(self.shape))[axis]
    out_s = ViewTracker.generate_view(optype, [self], axis=axis, keepdim=keepdim)
    return create_buffer(out_s, optype, Function(optype, self), ctx=[axis, keepdim])
    #return Buffer(ViewTracker.generate_view(optype, [self], axis=axis, keepdim=keepdim), optype, [self], ctx=[axis, keepdim])

  def reshape_op(self, optype:Ops, args) -> Buffer: 
    return Buffer(ViewTracker.generate_view(optype, self, args=args), optype, [self], ctx=args)

  # a Buffer is realized if its data is not None
  def realized(self:Buffer) -> bool: return self.data is not None

  def is_contiguous(self:Buffer) -> bool: 
    return all(self.strides[i+1] >= self.strides[i] for i in range(len(self.strides) - 1))

  def merge_reshape_into_e(buf:Buffer, optype, args) -> Buffer: 
    buf.reshapes = optype
    buf.shape = ViewTracker.generate_view(optype, buf, args=args)
    if optype == ReshapeOPS.TRANSPOSE: buf.strides = tuple([buf.strides[::-1][_] for _ in args])[::-1]
    else: buf.strides = ViewTracker.generate_strides(buf.shape)
    buf.ctx = args
    return buf

  @staticmethod
  def const_load(shape:Tuple[int, ...], arg:int) -> Buffer:
    _loadop = LoadOP(shape, LoadOPS.CONST, arg=arg)
    return Buffer(shape, optype = LoadOPS.CONST, ctx = _loadop)

  @staticmethod
  def rand_load(shape:Tuple[int, ...]) -> Buffer:
    _loadop = LoadOP(shape, LoadOPS.RAND)
    return Buffer(shape, optype=LoadOPS.RAND, ctx = _loadop)

  def _alloc(self): 
    if self.data is None: 
      if not isinstance(self.ctx, LoadOP): 
        self.data = LoadOP.alloc_raw(self.shape)
      else:
        self.data = LoadOPSAllocator[self.optype](self.ctx.shape, self.ctx.arg)

  def alloc(self):
    self._alloc()
    if self.children: 
      for buf in self.children:
        buf._alloc()

  # we should combine this with the old realize function that toposorts the non LoadOPS
  # need way of storing already generated kernels for reuse
  # this should be done in passes: 1. Frontend OPT pass 2. Alloc pass 3. Tokenization pass 4. codegen pass
  # fusing reshapes/transpose into ops is not an OPT, we need it to reduce shitty code from the codegenerator
  def realize(self) -> Buffer:
    if self.optype in LoadOPS: 
      self.alloc()
      return self

    for f in self.children:
      if f.optype not in LoadOPS:
        if not f._realized(): f.realize() 
    
    tokenizer = Tokenizer(self) 
    kernel = C_Codegen(tokenizer.op).kernel
    self.alloc() 
    ExecuteCProgram(kernel, self, tokenizer.op.reg).run()
    return self


  def _realized(self): return self.data is not None

  @staticmethod
  def read_load(data) -> Buffer: 
    if isinstance(data, (int, float)): 
      _loadop = LoadOP((1,), LoadOPS.READ)
      return Buffer((1,), optype=LoadOPS.READ, ctx =_loadop, data=data)
    elif isinstance(data, np.ndarray): 
      data.astype(np.float32) if data.dtype != np.float32 else data
      _loadop = LoadOP(data.shape, LoadOPS.READ)
      return Buffer(data.shape, optype=LoadOPS.READ, ctx=_loadop, data=data)
    elif isinstance(data, list): 
      _loadop = LoadOP((len(data),1), LoadOPS.READ)
      _bufcast = np.array(data).astype(np.float32)
      return Buffer(_bufcast.shape, optype=LoadOPS.READ, ctx=_loadop, data=_bufcast)
    else: 
      raise NotImplementedError

class ViewTracker: 
  @classmethod 
  def generate_strides(self, shape): 
    if isinstance(shape, int): return (0,)
    strides, shape = [1], shape[::-1]
    for x in range(0, len(shape)-1): 
      strides.append(shape[x] * strides[-1])
    strides = tuple(strd if shp != 1 else 1 for strd, shp in zip(strides, list(shape)))
    return strides

  @classmethod
  def generate_view(self, op:Union[BinaryOPS, UnaryOPS, ReshapeOPS, ShapeOPS], in_buffers:Buffer, **kwargs) -> Tuple[int, ...]:
    ReshapeOPHelpers = {
      ReshapeOPS.RESHAPE: self._reshape,
      ReshapeOPS.SLICE: self._slice,
      ReshapeOPS.TRANSPOSE: self._transpose,
      ReshapeOPS.PAD: self._pad,
      ReshapeOPS.CAST: self._cast,
    }

    if op in BinaryOPS:
      assert in_buffers[0].shape[1] == in_buffers[1].shape[0] if op == BinaryOPS.MATMUL else in_buffers[0].shape == in_buffers[1].shape
      out_s = (in_buffers[0].shape[0], in_buffers[1].shape[1]) if op == BinaryOPS.MATMUL else in_buffers[0].shape 
      return out_s
    elif op in UnaryOPS: 
      out_s = in_buffers[0].shape
      return out_s
    elif op in ShapeOPS:
      axis, keepdim = kwargs['axis'], kwargs['keepdim']
      if axis is None: out_s = (1,)
      else:
        nx = list(axis) if isinstance(axis, tuple) else [axis]
        l = list(in_buffers[0].shape)
        for j in nx: l[j] =0 
        out_s = tuple([i for i in l if i!=0]) if keepdim == False else tuple([1 if i == 0 else i for i in l])
      return out_s
    elif op in ReshapeOPS: 
      return ReshapeOPHelpers[op](in_buffers, kwargs)
    
  def _reshape(in_s: Buffer, kwargs: dict) -> Tuple[int, ...]:
    arg, in_s = list(kwargs['args']), list(in_s.shape)
    out_s = tuple(arg)
    if -1 in arg:
      idx = arg.index(-1)
      _cur = np.prod([j for j in arg if j != -1])
      arg[idx] = np.prod(in_s)//_cur
      out_s = tuple(arg)
    return out_s

  def _slice(in_s: Buffer, kwargs: dict) -> Tuple[int, ...]:
    arg = kwargs['args'][0] if not isinstance(kwargs['args'][0], int) else kwargs['args'][0]
    # TEMPORARY HACK
    # we shouldnt be executing the slice to have it done, we need to interate through each of the slices and then calculate the output shape
    # numpy has broadcasting rules for how slices can be reduced EG: (1,1,5,5) -> (1,9,9) im2col the (9,1) 2nd index and the (9,9)(9,9) 3rd and 4th get broadcasted
    out_s = np.empty(in_s.shape)[arg].shape
    out_s = (1,) if out_s == () else out_s
    return out_s 

  def _transpose(in_s: Buffer, kwargs: dict) -> Tuple[int, ...]:
    arg, in_s = list(kwargs['args']), list(in_s.shape)
    return tuple([in_s[i] for i in arg])

  def _pad(in_s: Buffer, kwargs: dict) -> Tuple[int, ...]:
    return tuple([i+j for i, j in zip([sum(list(j)) for j in list(kwargs['args'])], (list(in_s.shape)))])
    
  def _cast(in_s: Buffer, kwargs: dict) -> Tuple[int, ...]:
    return tuple(kwargs['args'])
