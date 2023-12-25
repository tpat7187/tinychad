from __future__ import annotations
import numpy as np 
from typing import Union, Tuple, Optional, List
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS, LoadOPS

class LoadOP: 
  __slots__ = "shape", "arg", "loadop"
  def __init__(self, shape, loadop, arg=None): 
    self.shape, self.arg, self.loadop = shape, arg, loadop 

  def __repr__(self): return str(self.loadop)

class Buffer: 
  __slots__ = "shape", "op", "children", "data", "ctx", "strides"
  def __init__(self, shape, op, children:Optional[List[Buffer]]=None, data:Optional[np.ndarray]=None, ctx=None): 
      self.shape, self.op, self.children, self.ctx, self.data = shape, op, children, ctx, data
      self.strides = ViewTracker.generate_strides(shape)

  @property 
  def dtype(self): return np.float32

  def __repr__(self): 
    return f"<{type(self).__name__}: op = <{self.op}>: [shape = {self.shape}, strides = {self.strides}]>"

  def binary_op(self, fxn, x:Buffer)     -> Buffer: return Buffer(ViewTracker.generate_view(fxn, [self, x]), fxn, [self, x])
  def unary_op(self, fxn)                -> Buffer: return Buffer(self.shape, fxn, [self])
  def shape_op(self, fxn, axis, keepdim) -> Buffer: return Buffer(ViewTracker.generate_view(fxn, [self], axis=axis, keepdim=keepdim), fxn, [self], ctx=[axis, keepdim])
  def reshape_op(self, fxn, args)        -> Buffer: return Buffer(ViewTracker.generate_view(fxn, self, args=args), fxn, [self], ctx=args)

  # a Buffer is realized if its data is not None
  def realized(self:Buffer) -> bool: return self.data is not None

  @staticmethod
  def const_load(shape:Tuple[int, ...], arg:int) -> Buffer:
    _loadop = LoadOP(shape, LoadOPS.CONST, arg=arg)
    return Buffer(shape, op = LoadOPS.CONST, ctx = _loadop)

  @staticmethod
  def rand_load(shape:Tuple[int, ...]) -> Buffer:
    _loadop = LoadOP(shape, LoadOPS.RAND)
    return Buffer(shape, op=LoadOPS.RAND, ctx = _loadop)

  @staticmethod
  def read_load(data) -> Buffer: 
    if isinstance(data, (int, float)): 
      _loadop = LoadOP((1,), LoadOPS.READ)
      return Buffer((1,), op=LoadOPS.READ, ctx =_loadop, data=data)
    elif isinstance(data, np.ndarray): 
      data.astype(np.float32) if data.dtype != np.float32 else data
      _loadop = LoadOP(data.shape, LoadOPS.READ)
      return Buffer(data.shape, op=LoadOPS.READ, ctx=_loadop, data=data)
    elif isinstance(data, list): 
      _loadop = LoadOP((len(data),1), LoadOPS.READ)
      _bufcast = np.array(data).astype(np.float32)
      return Buffer(_bufcast.shape, op=LoadOPS.READ, ctx=_loadop, data=_bufcast)
    else: 
      raise NotImplementedError

  def _alloc(self): 
    if self.op.loadop == LoadOPS.RAND: 
      return self.alloc_rand(self.op.shape)
    elif self.op.loadop == LoadOPS.CONST: 
      return self.alloc_const(self.op.shape, self.op.arg)
    elif self.op.loadop == LoadOPS.READ: 
      return 
    else: raise NotImplementedError

  def alloc_const(self, shape:Tuple[int, ...], arg:int) -> np.ndarray:
    self.data = np.full(shape, arg).astype(np.float32)

  def alloc_rand(self, shape:Tuple[int, ...]) -> np.ndarray:
    self.data =  np.random.randn(*shape).astype(np.float32)

class ViewTracker: 
  @classmethod 
  def generate_strides(self, shape): 
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
