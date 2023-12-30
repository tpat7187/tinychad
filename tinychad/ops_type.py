from __future__ import annotations
import os
from typing import Union
from enum import Enum, auto

class UnaryOPS(Enum): RELU = auto(); NEG = auto(); LOG = auto(); EXP = auto(); SQRT = auto();
class BinaryOPS(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MATMUL = auto(); CMP = auto();
class ShapeOPS(Enum): MAX = auto(); SUM = auto();
class ReshapeOPS(Enum): RESHAPE = auto(); SLICE = auto(); PAD = auto(); TRANSPOSE = auto(); CAST = auto();
class LoadOPS(Enum): LOAD = auto(); RAND = auto(); CONST = auto(); READ = auto();

Ops = Union[UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS, LoadOPS]
DEBUG = os.getenv("DEBUG") 

class TokenType(Enum): 
  FUNCSTART = auto(),     # args: fxn_name, num_inputs
  FUNCEND = auto(),
  LOOPSTART = auto(),     # args: num_iterations, num_increment
  LOOPSTOP = auto(),      # args: condition
  OP = auto(),            # args: type, children
  CMP = auto(),
  LOAD = auto(),          # args: loadidx, inputbuffer
  GLOBAL = auto(),        # args: storeidx
  LOCAL = auto(),         # args: storeidx
  ACCUMULATE = auto()     
