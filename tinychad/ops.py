from __future__ import annotations
import numpy as np
from tinychad.tensor import OP
from typing import Union
from enum import Enum, auto
from tinychad.buffers import Buffer
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS

class LOAD(OP): 
  def __init__(self, saved = None):
    self.arg = type(self).__name__
    self.saved = saved

# binary ops
class ADD(OP): 
  # buffers passed in, not tensors
  @staticmethod
  def forward(x:Buffer, y:Buffer) -> Buffer: 
    return x.binary_op(BinaryOPS.ADD, y)
  
  def backward(self, out_grad, out): 
    return out_grad, out_grad

class SUB(OP): 
  @staticmethod
  def forward(x:Buffer, y:Buffer) -> Buffer:
    return x.binary_op(BinaryOPS.SUB, y)
  
  def backward(self, out_grad, out): 
    return out_grad, -out_grad

class MUL(OP): 
  @staticmethod
  def forward(x:Buffer, y:Buffer) -> Buffer:
    return x.binary_op(BinaryOPS.MUL, y)

  def backward(self, out_grad, out):
    return out_grad * self.saved[1].data, out_grad*self.saved[0].data

class DIV(OP): 
  @staticmethod
  def forward(x:Buffer, y:Buffer) -> Buffer:
    return x.binary_op(BinaryOPS.DIV, y)
  
  def backward(self, out_grad, out):
    return (self.saved[1].data**-1) * out_grad, -(self.saved[0].data/self.saved[1].data**2)*out_grad

class MATMUL(OP): 
  @staticmethod
  def forward(x:Buffer, y:Buffer) -> Buffer:
    return x.binary_op(BinaryOPS.MATMUL, y)

  def backward(self, out_grad, out):
    return np.matmul(out_grad, self.saved[1].T().data), np.matmul(self.saved[0].T().data, out_grad)

# unary ops
class RELU(OP):
  @staticmethod
  def forward(x:Buffer) -> Buffer:
    return x.unary_op(UnaryOPS.RELU)

  def backward(self, out_grad, out):
    return (out > 0)*out_grad

class EXP(OP): 
  @staticmethod
  def forward(x:Buffer) -> Buffer:
    return x.unary_op(UnaryOPS.EXP)

  def backward(self, out_grad, out):
    return out * out_grad

class LOG(OP): 
  @staticmethod
  def forward(x:Buffer) -> Buffer:
    return x.unary_op(UnaryOPS.LOG)

  def backward(self, out_grad, out):
    return out_grad / self.saved[0].data

class NEG(OP): 
  @staticmethod
  def forward(x:Buffer) -> Buffer:
    return x.unary_op(UnaryOPS.NEG)

  def backward(self, out_grad, out): 
    return -1*out_grad

class SQRT(OP): 
  @staticmethod 
  def forward(x:Buffer) -> Buffer:
    return x.unary_op(UnaryOPS.SQRT)

  def backward(self, out_grad, out): 
    return (1 / 2 * out_grad**2)

# shape ops
class SUM(OP):
  @staticmethod
  def forward(x:Buffer, axis, keepdim) -> Buffer:
    return x.shape_op(ShapeOPS.SUM, axis, keepdim)

  def backward(self, out_grad, out):
    return np.broadcast_to(out_grad, self.saved[0].shape)
      
class MAX(OP): 
  @staticmethod
  def forward(x:Buffer, axis, keepdim) -> Buffer: 
    return x.shape_op(ShapeOPS.MAX, axis, keepdim)

  def backward(self, out_grad, out):
    axis, kd = self.ctx[0], self.ctx[1]
    if kd is False:
      out = np.expand_dims(out, axis=axis) if axis is not None else out
      out_grad = np.expand_dims(out_grad, axis=axis) if axis is not None else out_grad
    tt = 1.0 - (self.saved[0].data < np.broadcast_to(out, self.saved[0].shape)).astype(np.float32)
    exp = np.broadcast_to(tt.sum(axis=axis,keepdims=True), self.saved[0].shape)
    out = (tt / exp) * np.broadcast_to(out_grad, self.saved[0].shape)
    return out

# reshape ops
class RESHAPE(OP): 
  @staticmethod 
  def forward(x:Buffer, args) -> Buffer:
    return x.reshape_op(ReshapeOPS.RESHAPE, args)

  def backward(self, out_grad, out): 
    return out_grad.reshape(self.saved[0].shape)

class CAST(OP):
  @staticmethod 
  def forward(x:Buffer, args) -> Buffer:
    return x.reshape_op(ReshapeOPS.CAST, args)

  def backward(self, out_grad, out): 
    diff = len(out_grad.shape) - len(self.saved[0].shape)
    if diff > 0: out_grad = out_grad.sum(axis=tuple(np.arange(diff)))
    t = tuple([i for i, (a, b) in enumerate(zip(out_grad.shape, self.saved[0].shape)) if a != b])
    out_grad = out_grad.sum(axis = t, keepdims = True)
    return out_grad

class SLICE(OP):
  @staticmethod
  def forward(x:Buffer, args) -> Buffer:
    return x.reshape_op(ReshapeOPS.SLICE, args)
    '''
    out = x.data[tuple(*args)]
    return out if out.shape != () else [out]
    '''

  def backward(self, out_grad, out):
    arg = self.ctx[0]
    acc = np.zeros_like(self.saved[0].data)
    np.add.at(acc, *arg, out_grad)
    return acc

class PAD(OP): 
  @staticmethod 
  def forward(x: Buffer, args) -> Buffer:
    return x.reshape_op(ReshapeOPS.PAD, args)

  def backward(self, out_grad, out): 
    w = tuple([slice(i[0], j-i[1], None) for i, j in zip(*self.ctx, out.shape)])
    out = out_grad[w]
    return out

class TRANSPOSE(OP): 
  @staticmethod
  def forward(x:Buffer, args) -> Buffer:
    return x.reshape_op(ReshapeOPS.TRANSPOSE, args)

  def backward(self, out_grad, out): 
    return np.transpose(out_grad, np.argsort(*self.ctx))

 


