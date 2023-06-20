import numpy as np
from tinychad.tensor import OP

class LOAD(OP): 
  def __init__(self, saved = None):
    self.arg = type(self).__name__
    self.saved = saved

# self and out_grad
def _unbr(out_grad, saved):
  def _unbroadcast(out_grad,saved):
    return all((m==n) | (m==1) | (n==1) for n,m in \
    zip(out_grad.shape[::-1], saved.shape[::-1]))
  # if true, out_grad needs to be reshaped to saved
  if _unbroadcast(out_grad, saved): 
    if out_grad.shape == saved.shape:
      return out_grad
    else: 
      out_grad = out_grad.sum(axis=1, keepdims = True)
  return out_grad

# takes in input tensor and output shape
# checks if they're broadcastable
# if they are then does the transformation

# binary ops
class ADD(OP): 
  @staticmethod
  def forward(x, y): return x.data + y.data
  
  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad
    self.saved[1].grad += out_grad

class SUB(OP): 
  @staticmethod
  def forward(x, y): 
    return x.data - y.data
  
  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad
    self.saved[1].grad += out_grad

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

class MAX(OP): 
  @staticmethod
  def forward(x, axis, keepdim): 
    if axis is None: 
      return np.array([x.data.max(keepdims = keepdim)])
    else:
      return x.data.max(axis=axis, keepdims = keepdim)

  # TODO: fix this omegalul
  # set the index of the maximum value to 1
  def backward(self, out_grad, out):
    l = len(self.saved[0].data)
    ind = np.unravel_index(out.argmax(), self.saved[0].shape)
    self.saved[0].grad[ind] += 1 / l

# shapes

class RESHAPE(OP): 
  @staticmethod 
  def forward(x, *shape): return np.reshape(x.data, shape)

  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad.reshape(self.saved[0].shape)


# prett unary i guess
class CAST(OP):
  @staticmethod 
  def forward(x, y):
    return np.broadcast_to(x.data, y)

  def backward(self, out_grad, out): 
    shp, r = self.ctx, out_grad
    for j in range(len(out_grad.shape) - len(self.saved[0].shape)):
      r = r.sum(axis=0)
    if len(r.shape) == len(self.saved[0].shape):
      # TODO: is this cringe
      r = r.sum(axis=0, keepdims = True)
    self.saved[0].grad += r


''' 
BINARY OPS
IF THEY ARE NOT THE SAME SHAPE WE CHECK IF THEY ARE CASTBALE
IF THEY ARE CASTABLE WE FIND OUT WHICH ONE NEEDS TO BE CAST 
PERFORM CAST
PERFORM ADD, ADD SAVED needs to take in <CAST> and <OTHER> 

def add(self, x): return self.cast_ops(ops.ADD, x)
def cast_ops(self, fxn, x):

return tensor(ops.ADD.forward(x_s, y_s), op = ops.ADD(saved = [x_s, y_s]))

'''


