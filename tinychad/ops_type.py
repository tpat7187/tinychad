from __future__ import annotations
import os
from typing import Union, List, Optional, Tuple, Any
from enum import Enum, auto

class UnaryOPS(Enum): RELU = auto(); NEG = auto(); LOG = auto(); EXP = auto(); SQRT = auto();
class BinaryOPS(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MATMUL = auto(); CMP = auto();
class ShapeOPS(Enum): MAX = auto(); SUM = auto();
class ReshapeOPS(Enum): RESHAPE = auto(); SLICE = auto(); PAD = auto(); TRANSPOSE = auto(); CAST = auto();
class LoadOPS(Enum): LOAD = auto(); RAND = auto(); CONST = auto(); READ = auto();


Ops = Union[UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS, LoadOPS]

class FusedOp: 
  inputs:List[Any] 
  ops:List[Ops]

class TokenType(Enum): 
  FUNCTION = auto(),      # args: fxn_name, num_inputs
  FUNCEND = auto(),
  LOOPSTART = auto(),     # args: start, num_iterations, num_increment
  LOOPSTOP = auto(),      # args: condition
  OP = auto(),            # args: type, children
  CMP = auto(),           # args: type, children
  LOAD = auto(),          # args: loadidx, inputbuffer
  GLOBAL = auto(),        # args: storeidx, storename
  LOCAL = auto(),         # args: loadfrom, loadat
  ACC = auto(),           # args: accfrom
  DEFINE_ACC = auto(),    
  INDEX = auto(),         # args: buffer, idx
  LITERAL = auto(),       # args: item
