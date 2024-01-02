from __future__ import annotations
import numpy as np 
from typing import Union, List, Optional, Tuple
from enum import Enum, auto
from tinychad.ops_type import BinaryOPS, UnaryOPS, ReshapeOPS, ShapeOPS,TokenType
from tinychad.helpers import DEBUG
from tinychad.codegen import C_Codegen

''' 
exmaple program, adding tensor.randn(5,5) to tensor.randn(5,5)
no optimizations

Backend optimizations: 

Loop Unrolling

summing a 5x5 matrix with axis = None

2 FOR LOOPS
void SUM_5_7_0(float* buffer0, float* buffer1){ 
  for (int idx0 = 0; idx0 < 7; idx0++ ) { 
    float acc0 = 0; 
    // Sum over each row in the current column (5 rows)
    for (int idx1 = 0; idx1 < 5; idx1++) {
      acc0 += buffer0[idx1 * 7 + idx0]; 
    }
    buffer1[idx0] = acc0; 
  }
}

[1., 1., 1., 1., 1., 1., 1.],
[1., 1., 1., 1., 1., 1., 1.],
[1., 1., 1., 1., 1., 1., 1.],
[1., 1., 1., 1., 1., 1., 1.],
[1., 1., 1., 1., 1., 1., 1.]])

array([5., 5., 5., 5., 5.])
'''

class Token: 
  codegen: str = ""
  def __init__(self, _type:TokenType, args:Optional[List[Token]]=None, reg:str=None): 
    self.type, self.args, self.reg = _type, args, reg

  def __repr__(self): return f"TOKEN: {self.type} {self.args}"

# buffer_stream -> token_stream per buffer
class Tokenizer:
  def __init__(self, buf):
    self.buf = buf
    self.inputs:int = len(self.buf.children)
    self.buf_names:List[str] = [f"buffer{x}" for x in range(self.inputs+1)]
    self.loops:List[Token] = [] 
    self.token_stream:List[Token] = []
    self.local_loads:List[Token] = []
    self.local_stores:List[Token] = []
    self.in_s:Tuple[int, ...,] = []
    self.out_s:Tuple[int, ...,] = []

    self.input_args:List[str] = self.buf_names[:self.inputs]
    self.output_args:List[str] = self.buf_names[self.inputs:][0]

    self.out_s = self.buf.shape
    for i in self.buf.children:
      if self.in_s != i.shape and self.in_s is not None:
        self.in_s = [self.in_s] if not isinstance(self.in_s, list) else self.in_s
        self.in_s.append(i.shape)

    self.tokenize_buffer()

    if DEBUG: 
      for _ in self.token_stream: 
        print(_)

    self.kernel = C_Codegen(self.token_stream).kernel

  def tokenize_buffer(self):
    self.generate_function() 

    # if the axis is between the two we add a second loop, otherwise we start the acc

    if self.op in ShapeOPS: 
      self.axis = self.buf.ctx[0] 
      self.strides = self.buf.children[0].strides

      gbl = np.prod(self.out_s) if self.axis is not None else 0
      local_loops = 1 if self.axis is None else 2 if 0 < self.axis < len(self.in_s[0])-1 else 1

      if gbl != 0:
        self.tokenize_loop(0, gbl, 1)
      acc = self.tokenize_start_acc()

      for _ in range(local_loops):
        lcl = self.in_s[0][self.axis] if self.axis is not None else np.prod(self.in_s[0])
        self.tokenize_loop(0, lcl, 1)

      '''
      axis=0: idx0*1 + idx1*7
      axis=1: idx0*7 + idx1*1
      '''

      if self.axis is not None:
        tt = []
        reversed_strides = self.strides if self.axis == 0 else self.strides[::-1]
        for i in range(len(self.loops)):
          stride = reversed_strides[i]
          tt.append(f"{self.loops[i].reg}*{stride}")
        tt = ' + '.join(tt)
      else: 
        tt = self.loops[0].reg

      idx = self.index(self.input_args[0], tt)
      self.tokenize_acc(idx, acc)

      if gbl > 0: 
        self.e(Token(TokenType.LOOPSTOP))

      self.tokenize_store(acc, self.tokenize_literal(0) if self.axis is None else self.loops[0])

      self.e(Token(TokenType.LOOPSTOP))


    if self.op in BinaryOPS or self.op in UnaryOPS:
      st, iters, inc = 0, self.buf.size, 1
      self.tokenize_loop(st, iters, inc) 
      self.tokenize_operation(self.loops[0])
      for _ in self.loops:
        self.e(Token(TokenType.LOOPSTOP))

    self.e(Token(TokenType.FUNCEND))
    return self.token_stream

  def tokenize_load(self, load_from, load_at): 
    load_tok = Token(TokenType.LOAD, args=[load_from, load_at])
    self.local_loads.append(load_tok)
    load_tok.reg = f"local{len(self.local_loads)}"
    self.e(load_tok)
    return load_tok

  def index(self, buffer, index): 
    index_tok = Token(TokenType.INDEX, args=[buffer, index], reg = f"{buffer}[{index}]")
    return index_tok

  def tokenize_loop(self, st, iters, inc):
    loop_name = f"idx{len(self.loops)}"
    _tok = Token(TokenType.LOOPSTART, args = [st, iters, inc])
    _tok.reg, _tok.start, _tok.iters, _tok.inc = loop_name, st, iters, inc
    self.e(_tok)
    self.loops.append(_tok)
    return _tok
  
  def tokenize_literal(self, item:int): 
    _tok = Token(TokenType.LITERAL, args = item, reg = item)
    return _tok

  def tokenize_acc(self, acc_tok:Token, load_tok:Token): 
    _tok = Token(TokenType.ACC, args = [acc_tok, load_tok])
    self.e(_tok) 
    return _tok
  
  def tokenize_start_acc(self): 
    acc_tok = Token(TokenType.DEFINE_ACC)
    acc_tok.reg = f"acc0" 
    self.e(acc_tok)
    return acc_tok

  def e(self, token:Token): 
    self.token_stream.append(token)

  # compute operattion and store locally
  def tokenize_operation(self, _tok:Token=None): 
    children = [] 
    for x in range(self.inputs*_tok.args[2]):
      children.append(self.index(self.input_args[x], _tok.reg))

    op_tok = Token(TokenType.OP, args = [self.op, children]) 
    store_reg = f"b{len(self.local_stores)}"
    store_tok = Token(TokenType.LOCAL, args = op_tok, reg = store_reg)
    self.e(store_tok)

    self.tokenize_store(store_tok, _tok) 

  # will take reg of token and store it in another place
  def tokenize_store(self, _tok:Token, store_idx:Token):
    global_store = Token(TokenType.GLOBAL, args = [_tok, store_idx], reg = self.output_args)
    self.e(global_store)

  # return FUNCSTART TOKEN
  def generate_function(self): 
    op_name = self.buf.op
    self.op = op_name
    # TODO: add something to stop the matmul
    if op_name in UnaryOPS or op_name in BinaryOPS: 
      self.fxn_name  = f"{str(op_name.name)}{''.join(['_' + str(j) for j in self.buf.shape])}"
      _tok = Token(TokenType.FUNCSTART, args = [self.fxn_name, self.buf_names])
      self.e(_tok) 
    elif op_name in ShapeOPS:
      in_s, axis = self.buf.children[0].shape, self.buf.ctx[0]
      self.fxn_name = f"{str(op_name.name)}{''.join(['_' + str(j) for j in in_s])}{'_' + str(axis) if axis is not None else ''}"
      _tok = Token(TokenType.FUNCSTART, args = [self.fxn_name, self.buf_names])
      self.e(_tok) 


