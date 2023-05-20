import numpy as np

class OP: 
  def __init__(self, saved = None, ctx = None):
    self.arg = type(self).__name__
    self.saved = np.array(saved)
    self.ctx = ctx

class LOAD(OP): 
  def __init__(self, saved = None):
    self.arg = type(self).__name__
    self.saved = saved

# binary ops
class ADD(OP): 
  @staticmethod
  def forward(x, y): 
    return x.data + y.data
  
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
  def forward(x, axis):
    return np.array([x.data.sum(axis = axis)])

  # TODO: write broadcasting so that this will work when we write NLL
    '''
    basically numpy broadcasting rules and pytorch broadcasting rules are a little different
    we need to write our own broadcasting so that we can pass gradients back like: 
    [5x1 to 5x5] when we do ops dimension wise
    '''
  def backward(self, out_grad, out):
    self.saved[0].grad += out_grad 

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
    self.saved[0].grad += (1/out) * out_grad

class RESHAPE(OP): 
  @staticmethod 
  def forward(x, *shape): return np.reshape(x.data, shape)

  # how do reshapes change grad values
  def backward(self, out_grad, out): 
    self.saved[0].grad.reshape(out.shape)





