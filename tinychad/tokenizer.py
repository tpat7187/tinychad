from __future__ import annotations
import numpy as np 
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple, Any
from enum import Enum, auto
from tinychad.ops_type import BinaryOPS, UnaryOPS, ReshapeOPS, ShapeOPS, TokenType, Ops
from tinychad.helpers import DEBUG

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
    token_repr = f"{curr_prefix}{self.arg} {self.reg} "
    if self.ctx is not None: 
      token_repr += f"<ctx: {self.ctx}>\n"
    else:
      token_repr += "\n"
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
    self.in_s:Tuple[int, ...,] = []
    self.out_s:Tuple[int, ...,] = []

    self.input_args:List[str] = self.buf_names[:self.inputs]
    self.output_args:List[str] = self.buf_names[self.inputs:][0]
    
    self.open_loops:List[Token] = []
    self.open_ll:int = 0 
    self.open_acc:int = 0

    self.out_s = self.buf.shape
    for i in self.buf.children:
      if self.in_s != i.shape and self.in_s is not None:
        self.in_s = [self.in_s] if not isinstance(self.in_s, list) else self.in_s
        self.in_s.append(i.shape)

    self.fxn:Token = None

    self.tokenize_buffer()

    if DEBUG: print(self.fxn)

  def tokenize_buffer(self):
    self.fxn = self.generate_function() 

    # if the axis is between the two we add a second loop, otherwise we start the acc
    if self.op in ShapeOPS: 
      op_reduce = BinaryOPS.ADD if self.op == ShapeOPS.SUM else BinaryOPS.GTT
      self.axis = self.buf.ctx[0] 
      self.strides = self.buf.children[0].strides

      if self.axis is not None: 
        blocked = True if 0 < self.axis < len(self.in_s[0])-1 else False
      else: blocked = False

      if blocked:
        gbl_size = self.in_s[0][0]
      else:
        gbl_size = np.prod(self.out_s) if self.axis is not None else 0

      local_loops = 1 if self.axis is None else 2 if blocked else 1
      gbl = self.tokenize_loop(0, gbl_size, 1) if gbl_size else None
      acc = self.tokenize_start_acc(parent = gbl) if local_loops == 1 else None
      if local_loops == 1: 
        if self.axis is not None: 
          lcl = self.tokenize_loop(0, np.prod(self.in_s[0][self.axis]), 1, parent = gbl) 
        else: lcl = self.tokenize_loop(0, np.prod(self.in_s[0]), 1, parent = gbl) 
      elif local_loops > 1: 
        for _ in range(local_loops):
          self.tokenize_loop(0, self.in_s[0][::-1][_], 1, parent=self.open_loops[-1])
        lcl = self.open_loops[-1]
        acc = self.tokenize_start_acc(parent = self.open_loops[1])

      if self.axis is not None: 
        statements = []
        if blocked:
          shifted_strides = tuple(np.roll(self.strides, self.axis))
          for _ in range(len(self.open_loops)):
            statements.append(f"{self.open_loops[_].reg}*{shifted_strides[_]}")
        else:
          for _ in range(len(self.open_loops)):
            shifted_strides = tuple(np.roll(self.strides, _))
            statements.append(f"{self.open_loops[_].reg}*{shifted_strides[-self.axis]}")
        indx = ' + '.join(statements)
      else: 
        indx = lcl.reg

      st = self.local_store(self.tokenize_operation([acc.reg, self.index(self.input_args[0], indx)], op = op_reduce), acc.reg)
      self.e(lcl, st)

      if blocked: 
        statements = []
        for _ in range(len(self.open_loops)-1):
          statements.append(f"{self.open_loops[_].reg}*{self.strides[:len(self.open_loops)-1][::-1][_]}")
        blocked_idx = ' + '.join(statements)

      out_idx = 0 if self.axis is None else gbl.reg
      if gbl: 
        if blocked: 
          self.e(self.open_loops[1], self.global_store(st, self.index(self.output_args, blocked_idx)))
        else:
          self.e(gbl, self.global_store(st, self.index(self.output_args, out_idx)))
      else:
        self.e(self.fxn, self.global_store(st, self.index(self.output_args, out_idx)))

    if self.op in BinaryOPS or self.op in UnaryOPS:
      st, iters, inc = 0, self.buf.size, 1
      lcl = self.tokenize_loop(st, iters, inc) 
      children = [self.index(self.input_args[x], lcl.reg) for x in range(self.inputs*lcl.ctx[2])]
      st = self.local_store(self.tokenize_operation(children))
      self.e(lcl, st)
      self.e(lcl, self.global_store(st, self.index(self.output_args, lcl.reg)))

  def tokenize_loop(self, st:int, iters:int, inc:int, parent:Optional[Token]=None) -> Token:
    assert st >= 0 and iters > 0 
    loop_name = f"idx{len(self.open_loops)}"
    _tok = Token(arg=TokenType.LOOP, src = [], reg=loop_name, ctx=[st, iters, inc])
    self.open_loops.append(_tok)
    self.e(parent, _tok) if parent else self.e(self.fxn, _tok)
    return _tok

  def local_store(self, to_store:Token, register:Optional[Token]=None) -> Token:
    if not register: register = f'b{self.open_ll}'
    _tok = Token(arg=TokenType.LOCAL, src = [], reg=register) 
    self.e(_tok, to_store)
    return _tok

  def global_store(self, to_store:Token, store_reg:Token, index:Optional[Union[Token, int]]=None) -> Token:
    _tok = Token(arg=TokenType.GLOBAL, src = [store_reg, to_store.reg])
    return _tok

  def index(self, buffer, index) -> Token: 
    index_tok = Token(arg=TokenType.INDEX, src=[], reg=f"{buffer}[{index}]")
    return index_tok

  def e(self, parent:Token, child:Token, push:Optional[bool]=False): 
    if push: parent.src.insert(0, child)
    else:
      parent.src.append(child)

  def tokenize_operation(self, _in:List[Token], op:[Optional]=None) -> Token:
    op_tok = Token(arg=TokenType.OP, src = _in, reg = self.op if op is None else op) 
    return op_tok
  
  def tokenize_literal(self, item:int) -> Token:
    _tok = Token(arg=TokenType.LITERAL, src = item, reg = item)
    return _tok

  def tokenize_start_acc(self, parent:Optional[Token]=None) -> Token:
    acc_tok = Token(TokenType.DEFINE_ACC, src=[], reg=f"acc{self.open_acc}")
    self.e(parent, acc_tok, True) if parent else self.e(self.fxn, acc_tok)
    return acc_tok

  def generate_function(self) -> Token: 
    op_name = self.buf.op
    self.op = op_name
    # TODO: add something to stop the matmul
    if op_name in UnaryOPS or op_name in BinaryOPS: 
      fxn_name  = f"{str(op_name.name)}{''.join(['_' + str(j) for j in self.buf.shape])}"
      _tok = Token(arg=TokenType.FUNCTION, src=[], reg=fxn_name, ctx=self.buf_names)
    elif op_name in ShapeOPS:
      in_s, axis = self.buf.children[0].shape, self.buf.ctx[0]
      fxn_name = f"{str(op_name.name)}{''.join(['_' + str(j) for j in in_s])}{'_' + str(axis) if axis is not None else ''}"
      _tok = Token(arg=TokenType.FUNCTION, src = [], reg=fxn_name, ctx=self.buf_names)
    return _tok


