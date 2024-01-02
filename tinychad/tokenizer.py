from __future__ import annotations
import numpy as np 
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple, Any
from enum import Enum, auto
from tinychad.ops_type import BinaryOPS, UnaryOPS, ReshapeOPS, ShapeOPS, TokenType, Ops
from tinychad.helpers import DEBUG
from tinychad.codegen import C_Codegen

''' 
Backend optimizations: 

Loop Unrolling
Constant Folding

'''

@dataclass(repr=False)
class Token: 
  arg:TokenType
  src:List[Union[Token, int]]
  reg:Optional[str]=""
  ctx:Optional[Any]=None


  def __repr__(self, level=0, is_last=True, prefix=""):
    if level > 0:
      connector = "┗━ " if is_last else "┣━ "
    else:
      connector = ""
    curr_prefix = f"{prefix}{connector}"
    token_repr = f"{curr_prefix}{self.arg} {self.reg}\n"
    if level > 0:
      prefix += "    " if is_last else "┃   "
    for i, src_item in enumerate(self.src):
      is_last_child = (i == len(self.src) - 1)
      if isinstance(src_item, Token):
        token_repr += src_item.__repr__(level + 1, is_last_child, prefix)
      else:
        child_connector = "┗━ " if is_last_child else "┣━ "
        token_repr += f"{prefix}{child_connector}{src_item}\n"
    return token_repr


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
    
    self.open_loops:int = 0 
    self.open_ll:int = 0 

    self.out_s = self.buf.shape
    for i in self.buf.children:
      if self.in_s != i.shape and self.in_s is not None:
        self.in_s = [self.in_s] if not isinstance(self.in_s, list) else self.in_s
        self.in_s.append(i.shape)

    self.fxn:Token

    self.tokenize_buffer()

    if DEBUG: print(self.fxn)

    #self.kernel = C_Codegen(self.token_stream).kernel

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

      if self.axis is not None:
        tt = []
        reversed_strides = self.strides if self.axis == 0 else self.strides[::-1]
        for i in range(len(self.loops)):
          stride = reversed_strides[i]
          tt.append(f"{self.loops[i].reg}*{stride}")
        tt = ' + '.join(tt)

      idx = self.index(self.input_args[0], self.loops[0].reg if self.axis is None else tt)

      self.tokenize_acc(idx, acc)
      self.e(Token(TokenType.LOOPSTOP))

      if gbl > 0: 
        self.e(Token(TokenType.LOOPSTOP))

      self.tokenize_store(acc, self.tokenize_literal(0) if self.axis is None else self.loops[0])

    if self.op in BinaryOPS or self.op in UnaryOPS:
      st, iters, inc = 0, self.buf.size, 1
      lcl = self.tokenize_loop(st, iters, inc) 
      self.e(self.fxn, lcl)
      children = [self.index(self.input_args[x], lcl.reg) for x in range(self.inputs*lcl.ctx[2])]
      st = self.local_store(self.tokenize_operation(children))
      self.e(lcl, st)
      gbl = self.global_store(st, self.index(self.output_args, lcl.reg))
      self.e(lcl, gbl)

  def tokenize_loop(self, st, iters, inc):
    assert st >= 0 and iters > 0 
    loop_name = f"idx{self.open_loops}"
    _tok = Token(arg=TokenType.LOOPSTART, src = [], reg=loop_name, ctx=[st, iters, inc])
    self.open_loops +=1
    return _tok

  def local_store(self, to_store:Token): 
    local_register = f'b{self.open_ll}'
    _tok = Token(arg=TokenType.LOCAL, src = [], reg=local_register) 
    self.e(_tok, to_store)
    return _tok

  def global_store(self, to_store:Token, store_reg:Token, index:Optional[Union[Token, int]]=None): 
    _tok = Token(arg=TokenType.GLOBAL, src = [store_reg, to_store.reg])
    return _tok

  def index(self, buffer, index): 
    index_tok = Token(arg=TokenType.INDEX, src=[], reg=f"{buffer}[{index}]")
    return index_tok

  def e(self, parent:Token, child:Token): 
    parent.src.append(child)

  def tokenize_operation(self, _in:List[Token], store_reg:Optional[Token]=None):
    op_tok = Token(arg=TokenType.OP, src = _in, reg = self.op) 
    return op_tok

    '''
    if self.op in BinaryOPS or self.op in UnaryOPS:
      st, iters, inc = 0, self.buf.size, 1
      lcl = self.tokenize_loop(st, iters, inc) 

      children = [self.index(self.input_args[x], lcl.reg) for x in range(self.inputs*lcl.args[2])]
      op_tok = self.tokenize_operation(children)
      local = self.tokenize_local(op_tok)
      self.tokenize_store(local, lcl) 

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

  def tokenize_operation(self, reg:List[Token], store_reg:Optional[Token]=None):
    op_tok = Token(TokenType.OP, args = [self.op, reg]) 
    return op_tok
  '''

  # return FUNCSTART TOKEN
  def generate_function(self): 
    op_name = self.buf.op
    self.op = op_name
    # TODO: add something to stop the matmul
    if op_name in UnaryOPS or op_name in BinaryOPS: 
      fxn_name  = f"{str(op_name.name)}{''.join(['_' + str(j) for j in self.buf.shape])}"
      _tok = Token(arg=TokenType.FUNCTION, src=[], reg=fxn_name)
      self.fxn = _tok
    elif op_name in ShapeOPS:
      in_s, axis = self.buf.children[0].shape, self.buf.ctx[0]
      self.fxn_name = f"{str(op_name.name)}{''.join(['_' + str(j) for j in in_s])}{'_' + str(axis) if axis is not None else ''}"
      _tok = Token(TokenType.FUNCTION, args = [], reg=fxn_name)


