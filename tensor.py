import numpy as np 

#### OPS CLASS ####

class OP: 
  def __init__(self, saved = None):
    self.arg = arg
    self.saved = np.array(saved)

  def __repr__(self): 
    return self.arg

class ADD(OP): 
  def __init__(self, saved = None): 
    self.arg = '<ADD>' 
    self.saved = np.array(saved)

  def forward(x, y): 
    return tensor(x.data + y.data, op = ADD(saved = (x,y)))
  
  def backward(self, out_grad, out): 
    self.saved[0].grad += out_grad
    self.saved[1].grad += out_grad

class MATMUL(OP): 
  def __init__(self, saved = None): 
    self.arg = '<MATMUL>' 
    self.saved = np.array(saved)

  def forward(x, y): 
    return tensor(np.dot(x.data, y.data), op = MATMUL(saved = (x,y)))

  def backward(self, out_grad, out):
    self.saved[0].grad += np.dot(out_grad, self.saved[1].T().data)
    self.saved[1].grad += np.dot(self.saved[0].T().data, out_grad)

class LOAD(OP): 
  def __init__(self, saved = None):
    self.arg = '<LOAD>'
    self.saved = saved

#### TENSOR CLASS ####

class tensor: 
  def __init__(self, data, op = LOAD()):
    self.data = np.array(data, dtype = np.float32)
    self.op = op 
    self.grad = np.zeros(self.data.shape)

  def ones(*shape): return tensor(np.ones(*shape))

  def __repr__(self): 
    return f"OP = {self.op}: shape = {self.data.shape}" 

  def __add__(self,x): return self.add(x)
  def __matmul__(self,x): return self.dot(x)


  # fundamental OPS
  def add(self, x): return ADD.forward(self, x)
  def dot(self, x): return MATMUL.forward(self, x)


  # unary ops
  def T(self): return tensor(self.data.transpose())

  # i think this works
  def toposort(self): 
    topo, vis = [], set()
    def _toposort(s): 
      if s not in vis: 
        vis.add(s)
        if not isinstance(s.op, LOAD):
          for child in s.op.saved: 
            _toposort(child)
          topo.append(s)
    _toposort(self)
    return topo

  def backward(self): 
    self.grad = np.ones(self.grad.shape)
    for x in reversed(self.toposort()): 
      x.op.backward(x.grad, x.data)


def test(): 
  x = tensor.ones((5,5))
  y = tensor.ones((5,5))
  w = x @ y

  w.backward()

if __name__ == "__main__":
  test()



