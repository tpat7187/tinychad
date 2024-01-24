from __future__ import annotations
import numpy as np 
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple, Any
from enum import Enum, auto
from tinychad.ops_type import BinaryOPS, UnaryOPS, ReshapeOPS, ShapeOPS, TokenType, Ops, ControlType
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
  cond:Optional[Token]=None


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

    # if any children are not contiguous we perform opps elementwise ops by axis instead of by element
    self.contiguous_op = all(_.is_contiguous() for _ in self.buf.children)

    self.out_s = self.buf.shape
    for i in self.buf.children:
      if self.in_s != i.shape and self.in_s is not None:
        self.in_s = [self.in_s] if not isinstance(self.in_s, list) else self.in_s
        self.in_s.append(i.shape)
    if len(self.in_s)==1: self.in_s = self.in_s[0]

    self.fxn:Token = None

    self.tokenize_buffer()

    if int(DEBUG) > 1: print(self.fxn)

  def tokenize_buffer(self):
    self.fxn = self.generate_function() 

    # if the axis is between the two we add a second loop, otherwise we start the acc
    if self.op in ShapeOPS: 
      op_reduce = BinaryOPS.ADD if self.op == ShapeOPS.SUM else BinaryOPS.GTT
      self.axis = self.buf.ctx[0] 
      self.strides = self.buf.children[0].strides

      blocked = False if self.axis is None else 0 < self.axis < len(self.in_s)-1

      if blocked: 
        gbl_size = np.prod(self.in_s[:self.axis]) 
      else: 
        gbl_size = np.prod(self.out_s) if self.axis is not None else 0

      local_loops = 2 if blocked else 1 
      gbl = self.tokenize_loop(0, gbl_size, 1) if gbl_size else None
      acc = self.tokenize_start_acc(parent = gbl) if local_loops == 1 else None
      if local_loops == 1: 
        if self.axis is not None: 
          lcl = self.tokenize_loop(0, np.prod(self.in_s[self.axis]), 1, parent = gbl) 
        else: lcl = self.tokenize_loop(0, np.prod(self.in_s), 1, parent = gbl) 
      elif blocked:
          self.tokenize_loop(0, self.strides[::-1][self.axis], 1, parent=self.open_loops[-1])
          self.tokenize_loop(0, self.in_s[self.axis], 1, parent=self.open_loops[-1])
      lcl = self.open_loops[-1]
      acc = self.tokenize_start_acc(parent = self.open_loops[1]) if local_loops > 1 else acc

      if self.axis is not None: 
        statements = [] 
        if blocked: 
          # for indexing buffer0 
          indx = self.generate_shape_idx(self.open_loops)
          # for indexing buffer1
          blocked_idx = self.generate_shape_idx([self.open_loops[0], self.open_loops[1]])
        else: 
          for _ in range(len(self.open_loops)):
            shifted_strides = tuple(np.roll(self.strides, _))
            statements.append(self.tokenize_operation([self.open_loops[_].reg, shifted_strides[-self.axis]], BinaryOPS.MUL))
          indx = self.tokenize_operation([_ for _ in statements], BinaryOPS.ADD) 
      else: 
        indx = lcl.reg

      st = self.local_store(self.tokenize_operation([acc.reg, self.index(self.input_args[0], indx)], op = op_reduce), acc.reg, parent=lcl)

      if blocked: 
        self.e(self.open_loops[1], self.global_store(st, self.index(self.output_args, blocked_idx)))
      else:
        self.e(self.fxn if not gbl else gbl, self.global_store(st, self.index(self.output_args, 0 if self.axis is None else gbl.reg)))

    # TODO: refactor
    if self.op in BinaryOPS or self.op in UnaryOPS:
      # unary ops do not care about stride
      if not self.contiguous_op and self.op in BinaryOPS:
        src_stride = [_.strides[::-1] for _ in self.buf.children]
        for i in self.buf.shape:
          lcl = self.tokenize_loop(0, i, 1, nested=True)
        children = []
        for x in range(self.inputs*lcl.ctx[2]):
          st = self.MULACC(src_stride[x], self.open_loops)
          children.append(self.index(self.input_args[x], st))
      else:
        lcl = self.tokenize_loop(0, self.buf.size, 1) 
        children = [self.index(self.input_args[x], lcl.reg) for x in range(self.inputs*lcl.ctx[2])]
      st = self.local_store(self.tokenize_operation(children, self.op), parent=lcl)
      if self.contiguous_op: out_st = lcl.reg
      else: 
        out_st = self.MULACC(self.buf.strides[::-1], self.open_loops)
      self.global_store(st, self.index(self.output_args, out_st), parent=lcl)

    # we can probably repurpose this for PAD as they're both expand operations
    if self.op == ReshapeOPS.CAST:
      output_stride, input_stride = self.buf.strides, self.buf.children[0].strides[::-1]
      axis = [i for i,j in enumerate(self.out_s) if self.in_s[i] != j]
      [self.tokenize_loop(0, _, 1, nested=True) for _ in self.out_s] # can we do this in 2 loops every time
      output_st = self.MULACC(output_stride, reversed(self.open_loops))
      input_st = self.MULACC(input_stride, [j for i,j in enumerate(self.open_loops) if i not in axis])
      input_index, output_index = self.index(self.input_args[0], input_st), self.index(self.output_args, output_st)
      lcl = self.local_store(input_index, parent=self.open_loops[-1])
      gbl = self.global_store(lcl, output_index,parent=self.open_loops[-1])

    # TODO: combine this with CAST
    if self.op == ReshapeOPS.PAD:
      output_stride, input_stride = self.buf.strides, self.buf.children[0].strides[::-1]
      zv = tuple([(-b,s+e) for s,(b,e) in zip(self.in_s, self.buf.ctx)])
      mask = tuple([(b,s+b) for s,(b,_) in zip(self.in_s, self.buf.ctx)])
      offset = sum([s*p[0] for s,p in zip(input_stride, zv)])
      axis = [i for i,j in enumerate(self.out_s) if self.in_s[i] != j]

      self.tokenize_loop(0, self.out_s[0], 1, nested=True)
      self.tokenize_loop(0, np.prod(self.out_s[1:]), 1, nested=True)
      val = self.local_store(0, parent=self.open_loops[-1])
      # if ((ridx1 * (-1) < -1) && (ridx1 < 7) && (ridx0 * (-1) < -1) && (ridx0 < 7))
      pargs = [-(_[0]-1) for _ in self.buf.ctx]

      # n > 0 in (n, m)
      # m > 0 in (n, m) 
      nargs = [i for i, (f, _) in enumerate(self.buf.ctx) if f != 0]
      margs = [i for i, (_, f) in enumerate(self.buf.ctx) if f != 0]
      conds_p = [self.tokenize_operation([self.tokenize_operation([self.open_loops[_].reg, -1], BinaryOPS.MUL), pargs[_]], ControlType.LT) for _ in nargs]
      conds_e = [self.tokenize_operation([self.open_loops[j].reg, mask[-i][-1]], ControlType.LT) for i,j in enumerate(margs)]
      conds = conds_p + conds_e

      control = self.tokenize_control(conds, ControlType.AND, parent=self.open_loops[-1])
      input_st = self.tokenize_operation([self.MULACC(input_stride, self.open_loops), offset], BinaryOPS.ADD)
      input_index = self.index(self.input_args[0], input_st) 
      lcl = self.local_store(input_index, val.reg, parent=control) 

      output_st = self.MULACC(output_stride, reversed(self.open_loops))
      output_index = self.index(self.output_args, output_st)
      gbl = self.global_store(val, output_index,parent=self.open_loops[-1])

  def MULACC(self, tok:Token, args:List[Token]) -> Token: 
    inner = [self.tokenize_operation([j.reg, tok[i]], BinaryOPS.MUL) for i,j in enumerate(args)]
    out = self.tokenize_operation(inner, BinaryOPS.ADD) 
    return out
  
  # tokenIF -> result if true
  # cond will usually be a tokenize_operation
  def tokenize_control(self, cond:list[Token], ctx:TokenType=None, parent:bool=None): 
    _tok = Token(arg=TokenType.IF, src = [], ctx=ctx, cond=cond)
    if parent: self.e(parent, _tok) 
    return _tok

  def generate_shape_idx(self, loops:List[Token]) -> List[str]:
    assert len(loops) > 1
    len_loops, statements = len(loops), []
    for _ in range(len_loops-1):
      tt =[_ for _ in reversed(range(len_loops-1))]
      statements.append(self.tokenize_operation([loops[-_].reg, self.strides[::-1][self.axis-(tt[_])]], BinaryOPS.MUL))
    statements.append(self.open_loops[1].reg)
    return self.tokenize_operation([_ for _ in statements], BinaryOPS.ADD)

  def tokenize_loop(self, st:int, iters:int, inc:int, parent:Optional[Token]=None, nested:Optional[bool]=False) -> Token:
    assert st >=0 
    if iters == 0: return # dont run a loop if there are no iterations along that axis
    loop_name = f"idx{len(self.open_loops)}"
    _tok = Token(arg=TokenType.LOOP, src = [], reg=loop_name, ctx=[st, iters, inc])
    if parent: self.e(parent, _tok) 
    elif nested and len(self.open_loops) >=1: self.e(self.open_loops[-1], _tok)
    else: self.e(self.fxn, _tok)
    self.open_loops.append(_tok)
    return _tok

  def local_store(self, to_store:Token, register:Optional[Token]=None, parent:Optional[Token]=None) -> Token:
    if not register: register = f'b{self.open_ll}'
    _tok = Token(arg=TokenType.LOCAL, src = [], reg=register) 
    self.open_ll +=1
    self.e(_tok, to_store)
    if parent: self.e(parent, _tok) 
    return _tok

  def global_store(self, to_store:Token, store_reg:Token, parent:Optional[Token]=None) -> Token:
    _tok = Token(arg=TokenType.GLOBAL, src = [store_reg, to_store.reg])
    if parent: self.e(parent, _tok) 
    return _tok

  def index(self, buffer, index) -> Token: 
    index_tok = Token(arg=TokenType.INDEX, src=[buffer, index])
    return index_tok

  def e(self, parent:Token, child:Token, push:Optional[bool]=False): 
    if push: parent.src.insert(0, child)
    else:
      parent.src.append(child)

  # TODO: refactor
  def tokenize_operation(self, _in: List[Token], op:[Ops]) -> Token:
    if len(_in) == 1:
      if op in UnaryOPS: return Token(arg=TokenType.OP, src=[_in[0]], reg=op)
      else: return _in[0]
    if any(j == 1 for j in _in) and op == BinaryOPS.MUL: return _in[0]
    op_tok = Token(arg=TokenType.OP, src=[_in[0], _in[1]], reg=op)
    for token in _in[2:]:
        op_tok = Token(arg=TokenType.OP, src=[op_tok, token], reg=op)
    return op_tok
  
  def tokenize_literal(self, item:int) -> Token:
    _tok = Token(arg=TokenType.LITERAL, src = item, reg = item)
    return _tok

  def tokenize_start_acc(self, parent:Optional[Token]=None) -> Token:
    acc_tok = Token(TokenType.DEFINE_ACC, src=[], reg=f"acc{self.open_acc}", ctx = self.op)
    self.e(parent, acc_tok, True) if parent else self.e(self.fxn, acc_tok)
    self.open_acc += 1
    return acc_tok

  def generate_function(self) -> Token: 
    op_name = self.buf.op
    self.op = op_name
    # TODO: add something to stop the matmul
    if op_name in UnaryOPS or op_name in BinaryOPS: 
      fxn_name  = f"{str(op_name.name)}{''.join(['_' + str(j) for j in self.out_s])}"
      _tok = Token(arg=TokenType.FUNCTION, src=[], reg=fxn_name, ctx=self.buf_names)
    elif op_name in ShapeOPS:
      axis = self.buf.ctx[0]
      fxn_name = f"{str(op_name.name)}{''.join(['_' + str(j) for j in self.in_s])}{'_' + str(axis) if axis is not None else ''}"
      _tok = Token(arg=TokenType.FUNCTION, src = [], reg=fxn_name, ctx=self.buf_names)
    elif op_name in ReshapeOPS:
      fxn_name = f"{str(op_name.name)}{''.join(['_' + str(j) for j in self.in_s])}{''.join(['_' + str(j) for j in self.out_s])}"
      _tok = Token(arg=TokenType.FUNCTION, src = [], reg=fxn_name, ctx=self.buf_names)
    return _tok
