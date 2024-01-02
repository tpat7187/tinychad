from __future__ import annotations
import os, math, ctypes, subprocess, tempfile
import numpy as np 
from typing import Union, Tuple, Optional, List, Dict
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS, LoadOPS
from tinychad.tokenizer import Tokenizer
from tinychad.codegen import ExecuteCProgram

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

# maybe we just use buffers for kernel fusion and ast gen ;) 
class Buffer: 
  __slots__ = "shape", "op", "children", "data", "ctx", "strides"
  def __init__(self, shape, op, children:Optional[List[Buffer]]=None, data:Optional[np.ndarray]=None, ctx=None): 
      self.shape, self.op, self.children, self.ctx, self.data = shape, op, children, ctx, data

      self.strides = ViewTracker.generate_strides(shape)

  @property 
  def dtype(self): return np.float32

  @property
  def size(self): return math.prod(self.shape)

  def __repr__(self): 
    return f"<{type(self).__name__}: op = <{self.op}>: [shape = {self.shape}, strides = {self.strides}]>"


  def binary_op(self, fxn, x:Buffer) -> Buffer: return Buffer(ViewTracker.generate_view(fxn, [self, x]), fxn, [self, x])
  def unary_op(self, fxn) -> Buffer: return Buffer(self.shape, fxn, [self])
  def shape_op(self, fxn, axis, keepdim) -> Buffer: return Buffer(ViewTracker.generate_view(fxn, [self], axis=axis, keepdim=keepdim), fxn, [self], ctx=[axis, keepdim])
  def reshape_op(self, fxn, args) -> Buffer: return Buffer(ViewTracker.generate_view(fxn, self, args=args), fxn, [self], ctx=args)

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

  def _alloc(self): 
    if self.data is None: 
      if not isinstance(self.ctx, LoadOP): 
        self.data = LoadOP.alloc_raw(self.shape)
      else:
        self.data = LoadOPSAllocator[self.op](self.ctx.shape, self.ctx.arg)

  def alloc(self):
    self._alloc()
    if self.children: 
      for buf in self.children:
        buf._alloc()


  # has_cycle, takes in parent, and child and will assert that the other children cannot reach that particular child
  def merge_binary_ops(self, max_size: int = 5) -> Buffer:
    for i in self.children:
      if any(op in BinaryOPS for op in i.kernArgs) and self.has_cycle(i):
        print('fusing', self, i)
        self.kernArgs.extend(i.kernArgs)
        self.children.extend(i.children)
        self.children.remove(i)
        i.merge_binary_ops(max_size)

  def ast_kernel_fuser(self): 
    if OPT and self.children:
      if any(op in BinaryOPS for op in self.kernArgs): 
        self.merge_binary_ops()
      for child in self.children[:]:
        child.ast_kernel_fuser()

  # this is going to be really slow because for each op its going to need to check the entire computation graph below it
  def has_cycle(self, target, visited=None, rec_stack=None):
    if visited is None: 
      visited = set() 
    if rec_stack is None: 
      rec_stack = set()
    visited.add(self)
    rec_stack.add(self)
    if self.children:
      for child in self.children:
        if child == target and child in rec_stack:
            return True
        if child not in visited:
          if child.has_cycle(target, visited, rec_stack):
            return True
        elif child in rec_stack and child != target:
          continue
    rec_stack.remove(self)
    return False

  # we should combine this with the old realize function that toposorts the non LoadOPS
  # need way of storing already generated kernels for reuse
  def realize(self) -> Buffer:
    for f in self.children:
      if f.op not in LoadOPS:
        if not f._realized(): f.realize() 
    tok = Tokenizer(self) 
    self.alloc() 
    ExecuteCProgram(tok.kernel, self, tok.fxn_name).run()

  def _realized(self): return self.data is not None

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
