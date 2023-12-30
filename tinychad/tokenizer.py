import numpy as np 
from typing import Union
from enum import Enum, auto
from tinychad.ops_type import BinaryOPS, UnaryOPS, ReshapeOPS, ShapeOPS, DEBUG, TokenType
from tinychad.codegen import CPrinter

''' 
exmaple program, adding tensor.randn(5,5) to tensor.randn(5,5)
no optimizations

Backend optimizations: 

Loop Unrolling

void add_5_5(float* buffer1, float* buffer2, float* outbuffer) {
    for(int gidx = 0; gidx < 25; gidx++) {
          float b1 = buffer1[gidx];
          float b2 = buffer2[gidx];
          float acc0 = b1 + b2;
          outbuffer[gidx] = acc0;
    }
}

TOKENS: 
FUNCSTART
LOOPSTART
LOAD
LOAD
OP
GLOBAL
LOOPSTOP
FUNCEND
'''

# token, needs type, args
# generate all kernels
# each buffer will have a kern_object?
# kern_object will be a string
# toposort buffers, run kern_object in order
class Token: 
  def __init__(self, _type:TokenType, args = None): 
    self.type = _type
    self.args = args

    self.codegen = None

  def __repr__(self): return f"TOKEN: {self.type} {self.args}"

# buffer -> token_stream
class Tokenizer:
  def __init__(self, buf):
    self.buf = buf
    self.token_stream = []
    self.loop_count = 0
    self.inputs = len(self.buf.children)

    self.tokenize_buffer()
    self.kernel = CPrinter.generate_kernel(self)

  def tokenize_buffer(self):
    # all kernels have a function
    self.generate_function() 

    if self.op in BinaryOPS or self.op in UnaryOPS:
      # if LOOP UNROLLING, the step changes for example we do 5 ARITHS every loop
      # if loop is fully unrolled we skip adding a LOOPSTART 
      # inc is the number of ops that take place per iteration pretty much
      st, iters, inc = 0, self.buf.size, 1
      loop_name = f"idx{self.loop_count}"
      _tok = Token(TokenType.LOOPSTART, args = [st, iters, inc, loop_name])
      _tok.start, _tok.iters, _tok.inc = st, iters, inc
      self.token_stream.append(_tok)
      self.loop_count += 1
      self.tokenize_operation(_tok)

    for i in range(self.loop_count):
      self.token_stream.append(Token(TokenType.LOOPSTOP))
    self.token_stream.append(Token(TokenType.FUNCEND))

    return self.token_stream

  # will perform LOAD -> OP -> STORE inside the loop
  def tokenize_operation(self, _tok:Token=None): 
    local_loads, ops = [], 0
    for x in range(self.inputs*_tok.args[2]):
      load_tok = Token(TokenType.LOAD, args = [self.token_stream[0].args[1][x], _tok.args[3]])
      self.token_stream.append(load_tok)
      local_loads.append(load_tok)
      load_tok.reg = f"local{len(local_loads)}"

    op_tok = Token(TokenType.OP, args = [self.op, [_ for _ in local_loads]]) 
    op_tok.reg = f"acc{ops}" 
    ops += 1
    self.token_stream.append(op_tok)

    # storing the output 
    # TODO: make this more readable
    store_tok = Token(TokenType.GLOBAL, args = [self.token_stream[0].args[1][self.inputs], _tok.args[3]])
    store_tok.reg = op_tok.reg
    self.token_stream.append(store_tok)

  # return FUNCSTART TOKEN
  def generate_function(self): 
    op_name = self.buf.op
    self.op = op_name
    buf_names = [f"buffer{x}" for x in range(self.inputs+1)]
    # TODO: add something to stop the matmul
    if op_name in UnaryOPS or op_name in BinaryOPS: 
      self.in_s, self.out_s = self.buf.shape, self.buf.shape
      self.fxn_name  = f"{str(op_name.name)}{''.join(['_' + str(j) for j in self.in_s])}"
      _tok = Token(TokenType.FUNCSTART, args = [self.fxn_name, buf_names])
      self.token_stream.append(_tok) 


