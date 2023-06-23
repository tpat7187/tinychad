import numpy as np
from tinychad.tensor import OP

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


class MATMUL(OP): 
  @staticmethod
  def forward(x, y): 
    return np.dot(x.data, y.data)

  def backward(self, out_grad, out):
    self.saved[0].grad += np.dot(out_grad, self.saved[1].T().data)
    self.saved[1].grad += np.dot(self.saved[0].T().data, out_grad)

class MUL(OP): 
  @staticmethod
  def forward(x, y): 
    return x.data * y.data

  def backward(self, out_grad, out):
    self.saved[0].grad += self.saved[1].data * out_grad
    self.saved[1].grad += self.saved[0].data * out_grad


# do we need DIV
class DIV(OP): 
  @staticmethod
  def forward(x, y): 
    return x.data / y.data
  
  def backward(self, out_grad, out):
    self.saved[0].grad += (self.saved[1].data**-1) * out_grad
    self.saved[1].grad += -(self.saved[0].data/self.saved[1].data**2) * out_grad

# unary ops
class SUM(OP):
  @staticmethod
  def forward(x, axis, keepdim):
    return np.array([x.data.sum(keepdims = keepdim)]) if axis is None else x.data.sum(axis=axis, keepdims = keepdim)

  def backward(self, out_grad, out):
    if not isinstance(self.ctx, int):
      self.saved[0].grad += out_grad 
    else: 
      self.saved[0].grad += np.broadcast_to(out_grad, self.saved[0].grad.shape)

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
  def forward(x): 
    return np.log(x.data)

  def backward(self, out_grad, out):
    self.saved[0].grad += out_grad / self.saved[0].data

class NEG(OP): 
  @staticmethod
  def forward(x): return -1*x.data

  def backward(self, out_grad, out): 
    self.saved[0].grad += -1*out_grad

class MAX(OP): 
  @staticmethod
  def forward(x, axis, keepdim): 
    if axis is None: 
      return np.array([x.data.max(keepdims = keepdim)])
    else:
      return x.data.max(axis=axis, keepdims = keepdim)

  # when we MAX something we reshape it, by casting it to its original size and comparing with its input we can see which values match
  # matching values indicate where argmax found values along axis ; true -> 1, false -> 0
  def backward(self, out_grad, out):
    axis, kd = self.ctx[0], self.ctx[1]

    # broadcasting 'direction' changes depending on the axis we use
    if axis == 1:
      tt = np.broadcast_to(out.reshape(-1,1), self.saved[0].shape)
    else: 
      tt = np.broadcast_to(out, self.saved[0].shape)

    tt = (self.saved[0].data == tt).astype(np.promote_types(self.saved[0].data.dtype, tt.dtype))
    max_1s = tt

    expand = np.broadcast_to(max_1s.sum(axis=axis, keepdims = kd), self.saved[0].shape)
    max_amount = max_1s / expand

    grad_output_exp = np.broadcast_to(out_grad, self.saved[0].shape)
    self.saved[0].grad += max_amount * grad_output_exp

# shapes
class RESHAPE(OP): 
  @staticmethod 
  def forward(x, *shape):  return np.reshape(x.data, shape)



  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad.reshape(self.saved[0].shape)

# TODO: once the shapes are equal, we need to find the axis to sum over to make them the same
class CAST(OP):
  @staticmethod 
  def forward(x, y): return np.broadcast_to(x.data, y)

  def backward(self, out_grad, out): 
    # can we do this better? do we need a ctx

    shp, r = self.ctx, out_grad
    for j in range(len(out_grad.shape) - len(self.saved[0].shape)):
      r = r.sum(axis=0)
    if len(r.shape) == len(self.saved[0].shape) and len(r.shape)>1 and len(self.saved[0].shape)>1:
      ss = 0 
      for i,j in zip(r.shape, self.saved[0].shape): 
        if i != j: 
          break
        ss+=1
      r = r.sum(axis=ss, keepdims = True)
    self.saved[0].grad += r


''' 
BINARY OPS
IF THEY ARE NOT THE SAME SHAPE WE CHECK IF THEY ARE CASTBALE
IF THEY ARE CASTABLE WE FIND OUT WHICH ONE NEEDS TO BE CAST 
PERFORM CAST
PERFORM ADD, ADD SAVED needs to take in <CAST> and <OTHER> 
'''


