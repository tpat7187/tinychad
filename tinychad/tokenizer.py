import numpy as np 
from typing import Union
from enum import Enum, auto
from tinychad.ops_type import BinaryOPS, UnaryOPS, ReshapeOPS, ShapeOPS,TokenType
from tinychad.helpers import DEBUG
from tinychad.codegen import CPrinter

''' 
exmaple program, adding tensor.randn(5,5) to tensor.randn(5,5)
no optimizations

Backend optimizations: 

Loop Unrolling

[[1. 1. 1. 1.]
[1. 1. 1. 1.]
[1. 1. 1. 1.]]

[3. 3. 3. 3.]

void SUM_3_3_1(float* buffer0, float* buffer1) {
  for (int idx0 = 0; idx0 < 3; idx0++) {
    float acc0 = 0; 
    acc0 += buffer0[idx0 * 3];  // First element in the row
    acc0 += buffer0[idx0 * 3 + 1];  // Second element in the row
    acc0 += buffer0[idx0 * 3 + 2];  // Third element in the row
    buffer1[idx0] = acc0;  // Store the result in the corresponding position
  }
}
'''

# token, needs type, args
# generate all kernels
# each buffer will have a kern_object?
# kern_object will be a string
# toposort buffers, run kern_object in order
class Token: 


  def __init__(self, _type:TokenType, args=None, reg=None): 
    self.type = _type
    self.args = args
    self.reg = reg
    self.codegen = None

  def __repr__(self): return f"TOKEN: {self.type} {self.args}"

# buffer -> token_stream
class Tokenizer:
  def __init__(self, buf):
    self.buf = buf
    self.token_stream = []
    self.loops = []
    self.local_loads = []
    self.inputs = len(self.buf.children)

    self.tokenize_buffer()

    if DEBUG: 
      for _ in self.token_stream: 
        print(_)

    self.kernel = CPrinter.generate_kernel(self)

  def tokenize_buffer(self):
    self.generate_function() 

    # if the axis is between the two we add a second loop, otherwise we start the acc
    if self.op in ShapeOPS: 
      in_shape, axis = self.buf.children[0].shape, self.buf.ctx[0]
      self.tokenize_loop(0, np.prod(self.buf.shape), 1)
      acc = self.tokenize_start_acc()

      for _ in range(np.prod(self.buf.shape)):
        load_from = self.token_stream[0].args[1][0]
        load_at = f"{self.loops[0].reg}*{self.buf.children[0].strides[axis]}+{_}"
        self.tokenize_acc(self.tokenize_load(load_from,load_at), acc)

      store_at = self.token_stream[0].args[1][self.inputs]
      store_tok = Token(TokenType.GLOBAL, args=[store_at, self.loops[-1].reg], reg=acc.reg)
      self.token_stream.append(store_tok)

    if self.op in BinaryOPS or self.op in UnaryOPS:
      st, iters, inc = 0, self.buf.size, 1
      self.tokenize_loop(st, iters, inc) 
      self.tokenize_operation(self.loops[0])

    # this should be done manually
    for _ in self.loops:
      self.token_stream.append(Token(TokenType.LOOPSTOP))

    self.token_stream.append(Token(TokenType.FUNCEND))
    return self.token_stream

  def tokenize_load(self, load_from, load_at): 
    load_tok = Token(TokenType.LOAD, args=[load_from, load_at])
    self.local_loads.append(load_tok)
    load_tok.reg = f"local{len(self.local_loads)}"
    self.token_stream.append(load_tok)
    return load_tok

  def tokenize_loop(self, st, iters, inc):
    loop_name = f"idx{len(self.loops)}"
    _tok = Token(TokenType.LOOPSTART, args = [st, iters, inc])
    _tok.reg, _tok.start, _tok.iters, _tok.inc = loop_name, st, iters, inc
    self.token_stream.append(_tok)
    self.loops.append(_tok)
    return _tok

  def tokenize_acc(self, acc_tok, load_tok): 
    _tok = Token(TokenType.ACC, args = [acc_tok, load_tok])
    self.token_stream.append(_tok) 
    return _tok
  
  def tokenize_start_acc(self): 
    acc_tok = Token(TokenType.DEFINE_ACC)
    acc_tok.reg = f"acc0" 
    self.token_stream.append(acc_tok)
    return acc_tok

  # will perform LOAD -> OP -> STORE inside the loop
  def tokenize_operation(self, _tok:Token=None): 
    ops = 0 
    for x in range(self.inputs*_tok.args[2]):
      load_tok = Token(TokenType.LOAD, args = [self.token_stream[0].args[1][x], _tok.reg])
      self.token_stream.append(load_tok)
      self.local_loads.append(load_tok)
      load_tok.reg = f"local{len(self.local_loads)}"

    op_tok = Token(TokenType.OP, args = [self.op, [_ for _ in self.local_loads]]) 
    op_tok.reg = f"acc{ops}" 
    ops += 1
    self.token_stream.append(op_tok)

    # storing the output 
    # TODO: make this more readable
    store_tok = Token(TokenType.GLOBAL, args = [self.token_stream[0].args[1][self.inputs], _tok.reg])
    store_tok.reg = op_tok.reg
    self.token_stream.append(store_tok)

  # return FUNCSTART TOKEN
  def generate_function(self): 
    op_name = self.buf.op
    self.op = op_name
    buf_names = [f"buffer{x}" for x in range(self.inputs+1)]
    # TODO: add something to stop the matmul
    if op_name in UnaryOPS or op_name in BinaryOPS: 
      self.fxn_name  = f"{str(op_name.name)}{''.join(['_' + str(j) for j in self.buf.shape])}"
      _tok = Token(TokenType.FUNCSTART, args = [self.fxn_name, buf_names])
      self.token_stream.append(_tok) 
    elif op_name in ShapeOPS:
      in_s, axis = self.buf.children[0].shape, self.buf.ctx[0]
      self.fxn_name = f"{str(op_name.name)}{''.join(['_' + str(j) for j in in_s])}{'_' + str(axis) if axis is not None else ''}"
      _tok = Token(TokenType.FUNCSTART, args = [self.fxn_name, buf_names])
      self.token_stream.append(_tok) 


