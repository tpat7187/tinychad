from __future__ import annotations
import numpy as np
from tinychad.tensor import OP
from typing import Union
from tinychad.buffers import Buffer, Buffer
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS

# binary ops
class ADD(OP): 
  __slots__ = "x", "y"
  @staticmethod
  def forward(x:Buffer, y:Buffer) -> Buffer: 
    return x.binary_op(BinaryOPS.ADD, y)
  
  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out_grad, out_grad

class SUB(OP): 
  __slots__ = "x", "y"
  @staticmethod
  def forward(x:Buffer, y:Buffer) -> Buffer:
    return x.binary_op(BinaryOPS.SUB, y)
  
  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out_grad, -out_grad

class MUL(OP): 
  __slots__ = "x", "y"
  @staticmethod
  def forward(x:Buffer, y:Buffer) -> Buffer:
    return x.binary_op(BinaryOPS.MUL, y)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out_grad * self.saved[1].detach(), out_grad*self.saved[0].detach()

class DIV(OP): 
  __slots__ = "x", "y"
  @staticmethod
  def forward(x:Buffer, y:Buffer) -> Buffer:
    return x.binary_op(BinaryOPS.DIV, y)
  
  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return (self.saved[1].detach()**-1) * out_grad, -(self.saved[0].detach()/self.saved[1].detach()**2)*out_grad

class MATMUL(OP): 
  __slots__ = "x", "y"
  @staticmethod
  def forward(x:Buffer, y:Buffer) -> Buffer:
    return x.binary_op(BinaryOPS.MATMUL, y)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return np.matmul(out_grad, self.saved[1].detach().T), np.matmul(self.saved[0].detach().T, out_grad)

class GTT(OP): 
  __slots__ = "x", "y" 
  @staticmethod 
  def forward(x: Buffer, y:Buffer) -> Buffer:
    return x.binary_op(BinaryOPS.MAX, y)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out, out

# unary ops
class RELU(OP):
  __slots__ = "x"
  @staticmethod
  def forward(x:Buffer) -> Buffer:
    return x.unary_op(UnaryOPS.RELU)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return (out > 0)*out_grad

class EXP(OP): 
  __slots__ = "x"
  @staticmethod
  def forward(x:Buffer) -> Buffer:
    return x.unary_op(UnaryOPS.EXP)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out * out_grad

class LOG(OP): 
  __slots__ = "x"
  @staticmethod
  def forward(x:Buffer) -> Buffer:
    return x.unary_op(UnaryOPS.LOG)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out_grad / self.saved[0].detach()

class NEG(OP): 
  __slots__ = "x"
  @staticmethod
  def forward(x:Buffer) -> Buffer:
    return x.unary_op(UnaryOPS.NEG)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return -1*out_grad

class SQRT(OP): 
  __slots__ = "x"
  @staticmethod 
  def forward(x:Buffer) -> Buffer:
    return x.unary_op(UnaryOPS.SQRT)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return (1 / 2 * out_grad**2)

# shape ops
class SUM(OP):
  __slots__ = "x", "axis", "keepdim"
  @staticmethod
  def forward(x:Buffer, axis, keepdim) -> Buffer:
    return x.shape_op(ShapeOPS.SUM, axis, keepdim)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return np.broadcast_to(out_grad, self.saved[0].shape)
      
class MAX(OP): 
  __slots__ = "x", "axis", "keepdim"
  @staticmethod
  def forward(x:Buffer, axis, keepdim) -> Buffer: 
    return x.shape_op(ShapeOPS.MAX, axis, keepdim)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    axis, kd = self.ctx[0], self.ctx[1]
    if kd is False:
      out = np.expand_dims(out, axis=axis) if axis is not None else out
      out_grad = np.expand_dims(out_grad, axis=axis) if axis is not None else out_grad
    tt = 1.0 - (self.saved[0].detach() < np.broadcast_to(out, self.saved[0].shape)).astype(np.float32)
    exp = np.broadcast_to(tt.sum(axis=axis,keepdims=True), self.saved[0].shape)
    out = (tt / exp) * np.broadcast_to(out_grad, self.saved[0].shape)
    return out

# reshape ops
class RESHAPE(OP): 
  __slots__ = "x", "args"
  @staticmethod 
  def forward(x:Buffer, args) -> Buffer:
    return x.reshape_op(ReshapeOPS.RESHAPE, args)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return out_grad.reshape(self.saved[0].shape)

class CAST(OP):
  __slots__ = "x", "args"
  @staticmethod 
  def forward(x:Buffer, args) -> Buffer:
    return x.reshape_op(ReshapeOPS.CAST, args)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    diff = len(out_grad.shape) - len(self.saved[0].shape)
    if diff > 0: out_grad = out_grad.sum(axis=tuple(np.arange(diff)))
    t = tuple([i for i, (a, b) in enumerate(zip(out_grad.shape, self.saved[0].shape)) if a != b])
    out_grad = out_grad.sum(axis = t, keepdims = True)
    return out_grad

class SLICE(OP):
  __slots__ = "x", "args"
  @staticmethod
  def forward(x:Buffer, args) -> Buffer:
    return x.reshape_op(ReshapeOPS.SLICE, args)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    arg = self.ctx[0]
    acc = np.zeros_like(self.saved[0].detach())
    np.add.at(acc, *arg, out_grad)
    return acc

class PAD(OP): 
  __slots__ = "x", "args"
  @staticmethod 
  def forward(x: Buffer, args) -> Buffer:
    return x.reshape_op(ReshapeOPS.PAD, args)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    w = tuple([slice(i[0], j-i[1], None) for i, j in zip(*self.ctx, out.shape)])
    out = out_grad[w]
    return out

class TRANSPOSE(OP): 
  __slots__ = "x", "args"
  @staticmethod
  def forward(x:Buffer, args) -> Buffer:
    return x.reshape_op(ReshapeOPS.TRANSPOSE, args)

  def backward(self:OP, out_grad:np.ndarray, out:np.ndarray) -> np.ndarray:
    return np.transpose(out_grad, np.argsort(*self.ctx))

 


