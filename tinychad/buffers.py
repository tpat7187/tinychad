from __future__ import annotations
import numpy as np 
from typing import Union, Tuple, Optional, List
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS, LoadOPS, Compiled, Interpreted
from tinychad.llvm import LLVMCodegen

op_map = { 
    BinaryOPS.ADD   : lambda x, y: np.add(x,y),
    BinaryOPS.SUB   : lambda x, y: np.subtract(x,y),
    BinaryOPS.MUL   : lambda x, y: np.multiply(x,y),
    BinaryOPS.DIV   : lambda x, y: np.divide(x,y),
    BinaryOPS.MATMUL: lambda x, y: np.matmul(x,y),

    UnaryOPS.RELU   : lambda x : np.maximum(x, 0),
    UnaryOPS.EXP    : lambda x : np.exp(x),
    UnaryOPS.LOG    : lambda x : np.log(x),
    UnaryOPS.NEG    : lambda x : np.negative(x),
    UnaryOPS.SQRT   : lambda x : np.sqrt(x),

    ShapeOPS.MAX    : lambda x, axis, keepdims : np.max(x, axis, keepdims=keepdims),
    ShapeOPS.SUM    : lambda x, axis, keepdims : np.sum(x, axis, keepdims=keepdims), 

    ReshapeOPS.CAST : lambda x, args : np.broadcast_to(x, args),
    ReshapeOPS.PAD  : lambda x, args : np.pad(x, pad_width = args, mode = 'constant'), 
    ReshapeOPS.TRANSPOSE : lambda x, args : np.transpose(x, args),
    ReshapeOPS.RESHAPE : lambda x, args : np.reshape(x, args)
}

class Buffer: 
    __slots__ = "data", "shape", "strides", "op"
    def __init__(self, data:Union[np.ndarray, list, float, np.float32], op):
        if isinstance(data, np.ndarray):
            self.data = data

        if isinstance(data, (int, float, np.float32)):
            self.data = np.array([data], dtype=np.float32)

        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        
        self.op = op
        self.shape = self.data.shape
        self.strides = ViewTracker.generate_strides(self.shape)

    def __repr__(self): return f"{self.shape} Buffer"

    def realized(self): return True

    @property 
    def dtype(self): return self.data.dtype

    def __add__(self, x:Buffer) -> Buffer: return self.binary_op(BinaryOPS.ADD, x)
    def __sub__(self, x:Buffer) -> Buffer: return self.binary_op(BinaryOPS.SUB, x)
    def __mul__(self, x:Buffer) -> Buffer: return self.binary_op(BinaryOPS.MUL, x)
    def __div__(self, x:Buffer) -> Buffer: return self.binary_op(BinaryOPS.DIV, x)
    def __matmul__(self, x:Buffer) -> Buffer: return self.binary_op(BinaryOPS.MATMUL, x)

    def exp(self): return self.unary_op(UnaryOPS.EXP)
    def log(self): return self.unary_op(UnaryOPS.LOG)
    def neg(self): return self.unary_op(UnaryOPS.NEG)
    def relu(self): return self.unary_op(UnaryOPS.RELU)
    def sqrt(self): return self.unary_op(UnaryOPS.SQRT)

    # fxn example: BinaryOPS.ADD
    def binary_op(self, fxn, x): return op_map[fxn](self.data, x.data)
    def unary_op(self, fxn): return op_map[fxn](self.data)
    def shape_op(self, fxn, axis, keepdim): return op_map[fxn](self.data, axis=axis, keepdims=keepdim)
    def reshape_op(self, fxn, args): 
        if fxn == ReshapeOPS.PAD: assert isinstance(args, (tuple, list))
        if fxn == ReshapeOPS.SLICE: return self.slice(args)
        return op_map[fxn](self.data, args)
    
    def slice(x:Buffer, args) -> Buffer:
        args = (args) if isinstance(args, int) else args
        out = x.data[tuple(*args)]
        return out if out.shape != () else [out]


# new lazy buffer
class LazyBuffer: 
    __slots__ = "shape", "op", "children", "data", "ctx", "strides", "backend"
    def __init__(self, shape, op, children:Optional[List[LazyBuffer]]=None, data:Optional[np.ndarray]=None, ctx=None, backend=None): 
        self.shape, self.op, self.children = shape, op, children
        self.ctx = ctx
        if data is None: 
           self.data = data
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, (int, float, np.float32)):
            self.data = np.array([data], dtype=np.float32)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)

        self.strides = ViewTracker.generate_strides(shape)
        self.backend = backend

    @property 
    def dtype(self): return np.float32

    def binary_op(self, fxn, x:LazyBuffer) -> LazyBuffer: 
       return LazyBuffer(ViewTracker.generate_view(fxn, [self, x]), fxn, [self, x])
    def unary_op(self, fxn) -> LazyBuffer: 
       return LazyBuffer(self.shape, fxn, [self])
    def shape_op(self, fxn, axis, keepdim) -> LazyBuffer: 
       return LazyBuffer(ViewTracker.generate_view(fxn, [self], axis=axis, keepdim=keepdim), fxn, [self], ctx=[axis, keepdim])
    def reshape_op(self, fxn, args) -> LazyBuffer: 
       return LazyBuffer(ViewTracker.generate_view(fxn, self, args=args), fxn, [self], ctx=args)

    # a LazyBuffer is realized if its data is not None
    def realized(self:LazyBuffer) -> bool: return self.data is not None

    def exec(self:LazyBuffer) -> Buffer:
      if self.backend in Interpreted:
        for f in self.children:
           if not f.realized(): f.exec()
        if self.op in (BinaryOPS or UnaryOPS):
          s = op_map[self.op](*[j.data for j in self.children])
        elif self.op in ShapeOPS:
          axis, keepdim = self.ctx
          s = op_map[self.op](*[j.data for j in self.children], axis=axis, keepdims=keepdim)
        elif self.op in ReshapeOPS:
          args = self.ctx
          s = op_map[self.op](self.data, args)
        return Buffer(s, self.op)
      else: 
        codegen = LLVMCodegen(self.get_buffers())
        codegen.compile()

        return Buffer(self.data, self.op)

    def toposort(self) -> Tuple[LazyBuffer, ...]: 
      topo, vis = [], []
      def _toposort(s: Buffer):
        if s not in vis: 
          vis.append(s)
          if s.op != LoadOPS.LOAD:
            for child in s.children: 
              _toposort(child)
            topo.append(s)
      _toposort(self)
      return topo

    def get_buffers(self) -> Tuple: 
      cache, loads = [], []
      for s in self.toposort():
        for i in s.children:
          if i.op == LoadOPS.LOAD:
            loads.append((hex(id(i)), i.shape, i.op, i))
        _saved = tuple([hex(id(f)) for f in s.children])
        if isinstance(s, LazyBuffer): 
          s.data = np.zeros(s.shape, dtype=np.float32)
        _reg = (hex(id(s)), s.shape, s.op, s) + _saved
        cache.append(_reg)
      return loads + cache


# this may just be a LazyBuffer with a different codegen module
class GPUBuffer: 
    def __init__(self, shape, op):
        pass

class ViewTracker: 
  @classmethod 
  def generate_strides(self, shape): 
    strides, shape = [1], shape[::-1]
    for x in range(0, len(shape)-1): 
      strides.append(shape[x] * strides[-1])
    strides = tuple(strd if shp != 1 else 1 for strd, shp in zip(strides, list(shape)))
    return strides

  @classmethod
  def generate_view(self, op:Union[BinaryOPS, UnaryOPS, ReshapeOPS, ShapeOPS], in_buffers:LazyBuffer, **kwargs) -> Tuple[int, ...]:
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
    
  def _reshape(in_s: LazyBuffer, kwargs: dict) -> Tuple[int, ...]:
    arg, in_s = list(kwargs['args']), list(in_s.shape)
    out_s = tuple(arg)
    if -1 in arg:
      idx = arg.index(-1)
      _cur = np.prod([j for j in arg if j != -1])
      arg[idx] = np.prod(in_s)//_cur
      out_s = tuple(arg)
    return out_s

  def _slice(in_s: LazyBuffer, kwargs: dict) -> Tuple[int, ...]:
    arg = kwargs['args'][0] if not isinstance(kwargs['args'][0], int) else kwargs['args'][0]
    # TEMPORARY HACK 
    # we shouldnt be executing the slice to have it done, we need to interate through each of the slices and then calculate the output shape
    # numpy has broadcasting rules for how slices can be reduced EG: (1,1,5,5) -> (1,9,9) im2col the (9,1) 2nd index and the (9,9)(9,9) 3rd and 4th get broadcasted
    out_s = np.empty(in_s.shape)[arg].shape
    out_s = (1,) if out_s == () else out_s
    return out_s 

  def _transpose(in_s: LazyBuffer, kwargs: dict) -> Tuple[int, ...]:
    arg, in_s = list(kwargs['args']), list(in_s.shape)
    return tuple([in_s[i] for i in arg])

  def _pad(in_s: LazyBuffer, kwargs: dict) -> Tuple[int, ...]:
    return tuple([i+j for i, j in zip([sum(list(j)) for j in list(kwargs['args'])], (list(in_s.shape)))])
    
  def _cast(in_s: LazyBuffer, kwargs: dict) -> Tuple[int, ...]:
    return tuple(kwargs['args'])
