import numpy as np
from tinychad.tensor import OP
from enum import Enum, auto

class UnaryOPS(Enum): RELU = auto(); NEG = auto(); LOG = auto(); EXP = auto(); SQRT = auto();
class BinaryOPS(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MATMUL = auto(); 
class ShapeOPS(Enum): MAX = auto(); SUM = auto();
class ReshapeOPS(Enum): RESHAPE = auto(); SLICE = auto(); PAD = auto(); ROLL = auto(); TRANSPOSE = auto();

class LOAD(OP): 
  def __init__(self, saved = None):
    self.arg = type(self).__name__
    self.saved = saved

# binary ops
class ADD(OP): 
  @staticmethod
  def forward(x, y): return x.data + y.data
  
  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad
    self.saved[1].grad += out_grad

class SUB(OP): 
  @staticmethod
  def forward(x, y): return x.data - y.data
  
  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad
    self.saved[1].grad += -out_grad

class MUL(OP): 
  @staticmethod
  def forward(x, y): return x.data * y.data

  def backward(self, out_grad, out):
    self.saved[0].grad += self.saved[1].data * out_grad
    self.saved[1].grad += self.saved[0].data * out_grad

class DIV(OP): 
  @staticmethod
  def forward(x, y): return x.data / y.data
  
  def backward(self, out_grad, out):
    self.saved[0].grad += (self.saved[1].data**-1) * out_grad
    self.saved[1].grad += -(self.saved[0].data/self.saved[1].data**2) * out_grad

class MATMUL(OP): 
  @staticmethod
  def forward(x, y): return np.matmul(x.data, y.data)

  def backward(self, out_grad, out):
    self.saved[0].grad += np.matmul(out_grad, self.saved[1].T().data)
    self.saved[1].grad += np.matmul(self.saved[0].T().data, out_grad)

# unary ops
class RELU(OP):
  @staticmethod
  def forward(x): return np.maximum(x.data, 0)

  def backward(self, out_grad, out):
    self.saved[0].grad += (out > 0) *out_grad

class EXP(OP): 
  @staticmethod
  def forward(x): return np.exp(x.data)

  def backward(self, out_grad, out):
    self.saved[0].grad += out * out_grad

class LOG(OP): 
  @staticmethod
  def forward(x): return np.log(x.data)

  def backward(self, out_grad, out):
    self.saved[0].grad += out_grad / self.saved[0].data

class NEG(OP): 
  @staticmethod
  def forward(x): return -1*x.data

  def backward(self, out_grad, out): 
    self.saved[0].grad += -1*out_grad

class SQRT(OP): 
  @staticmethod 
  def forward(x): return np.sqrt(x.data)

  def backward(self, out_grad, out): 
    self.saved[0].grad += (1 / 2 * out_grad**2)

# shape ops
class SUM(OP):
  @staticmethod
  def forward(x, axis, keepdim):
    return np.array([x.data.sum(keepdims = keepdim)]) if axis is None else x.data.sum(axis=axis, keepdims = keepdim)

  def backward(self, out_grad, out):
    if not isinstance(self.ctx, int):
      self.saved[0].grad += out_grad 
    else: 
      self.saved[0].grad += np.broadcast_to(out_grad, self.saved[0].grad.shape)
      
class MAX(OP): 
  @staticmethod
  def forward(x, axis, keepdim): 
    if axis is None: 
      return np.array([x.data.max(keepdims = keepdim)])
    else:
      return x.data.max(axis=axis, keepdims = keepdim)

  # TODO: refactor for axis = N
  def backward(self, out_grad, out):
    axis, kd = self.ctx[0], self.ctx[1]
    if axis == 1:
      tt = np.broadcast_to(out.reshape(-1,1), self.saved[0].shape)
    else: 
      tt = np.broadcast_to(out, self.saved[0].shape)
    tt = (self.saved[0].data == tt).astype(np.promote_types(self.saved[0].data.dtype, tt.dtype))
    expand = np.broadcast_to(tt.sum(axis=axis, keepdims = kd), self.saved[0].shape)
    max_amount = tt / expand
    grad_output_exp = np.broadcast_to(out_grad, self.saved[0].shape)
    self.saved[0].grad += max_amount * grad_output_exp

# reshape ops
class RESHAPE(OP): 
  @staticmethod 
  def forward(x, *shape):  return np.reshape(x.data, shape)

  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad.reshape(self.saved[0].shape)

class CAST(OP):
  @staticmethod 
  def forward(x, y): return np.broadcast_to(x.data, y)

  def backward(self, out_grad, out): 
    shp, r, ss = self.ctx, out_grad, 0 
    diff = len(out_grad.shape) - len(self.saved[0].shape)
    if diff > 0: r = r.sum(axis=tuple(np.arange(diff)))
    t = tuple([i for i, (a, b) in enumerate(zip(r.shape, self.saved[0].shape)) if a != b])
    r = r.sum(axis = t, keepdims = True)
    self.saved[0].grad += r

# we support LOCAL slicing [x,y,z] NOT [x][y][z] idk if this is bad 
class SLICE(OP):
  @staticmethod
  def forward(x, args): 
    return x.data[args] if isinstance(args, (slice, tuple)) else np.array([x.data[args]])

  def backward(self, out_grad, out):
    arg = self.ctx[0]
    # accumulate gradients ; validate that this works for small slices?
    acc = np.zeros_like(self.saved[0].grad)
    np.add.at(acc, arg, out_grad)
    self.saved[0].grad += acc

class PAD(OP): 
  @staticmethod 
  def forward(x, args):
    assert isinstance(args, (tuple, list))
    out = np.pad(x.data, pad_width=args, mode='constant')
    return out

  def backward(self, out_grad, out): 
    w = tuple([slice(i[0], j-i[1], None) for i, j in zip(self.ctx, out.shape)])
    self.saved[0].grad += out_grad[w]

# can we get rid of this 
class ROLL(OP): 
  @staticmethod
  def forward(x, shift, axis): 
    return np.roll(x.data, shift, axis)

  def backward(self, out_grad, out): 
    shift, axis = self.ctx
    self.saved[0].grad += np.roll(out_grad, -shift, axis)

class TRANSPOSE(OP): 
  @staticmethod
  def forward(x, order): return np.transpose(x.data, order)

  def backward(self, out_grad, out): 
    self.saved[0].grad += np.transpose(out_grad, np.argsort(self.ctx))

 


