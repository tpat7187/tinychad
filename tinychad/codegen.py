from __future__ import annotations
import ctypes, subprocess, tempfile
from typing import Union, Tuple, Optional, List, Dict
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS, LoadOPS, TokenType, ControlType
from tinychad.helpers import DEBUG
from tinychad.tokenizer import Token
import numpy as np 

class ExecuteCProgram:
  def __init__(self, prg:str, bufs, fxn_name: str): 
    self.prg = prg
    self.args = [np.ctypeslib.as_ctypes(child.data) for child in bufs.children] + [np.ctypeslib.as_ctypes(bufs.data)]

    self.fxn_name = fxn_name
    self.dll = self.compile()

  def compile(self) -> ctypes.CDLL:
    with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as fp:
      subprocess.check_output(
        ['clang', '-shared', '-march=native', '-O3', '-Wall', '-Werror',
          '-x', 'c', '-fPIC', '-o', fp.name, '-'],
        input=self.prg.encode('utf-8')
      )
      lib = ctypes.CDLL(fp.name)
      return lib

  def run(self) -> None: 
    cfun = getattr(self.dll, self.fxn_name)
    cfun(*self.args)


class C_Codegen:  
  KERNEL_HEADER = "#import <math.h>\n#define max(x,y) ((x) >= (y)) ? (x) : (y)\n#define relu(x) (x > 0 ? x : 0)\n"

  op_map = { 
    BinaryOPS.ADD: lambda tok1, tok2: f"{tok1} + {tok2}",
    BinaryOPS.SUB: lambda tok1, tok2: f"{tok1} - {tok2}",
    BinaryOPS.MUL: lambda tok1, tok2: f"{tok1} * {tok2}",
    BinaryOPS.DIV: lambda tok1, tok2: f"{tok1} / {tok2}",
    BinaryOPS.GTT: lambda tok1, tok2: f"max({tok1}, {tok2})",
    UnaryOPS.LOG: lambda tok1: f"log({tok1})",
    UnaryOPS.EXP: lambda tok1: f"exp({tok1})",
    UnaryOPS.SQRT: lambda tok1: f"sqrt({tok1})",
    UnaryOPS.RELU: lambda tok1: f"relu({tok1})",
    UnaryOPS.NEG: lambda tok1: f"-{tok1}",
    ControlType.GT: lambda tok1, tok2: f"{tok1} > {tok2}",
    ControlType.LT: lambda tok1, tok2: f"{tok1} < {tok2}",
    ControlType.EQ: lambda tok1, tok2: f"{tok1} == {tok2}",
    ControlType.NEQ: lambda tok1, tok2: f"{tok1} != {tok2}"
  }

  def __init__(self, fxn_token): 
    self.fxn_token = fxn_token
    self.lines:List[str] = [self.KERNEL_HEADER]
    self.loads = {}
    self.kernel = self.generate_kernel(fxn_token)

    if int(DEBUG) > 2: 
      print(self.kernel)

  def generate_kernel(self, token):
    if not isinstance(token, Token): 
      return token

    if token.arg == TokenType.FUNCTION:
      cg = f"void {token.reg}({', '.join(['float* ' + _ for _ in token.ctx])}) {{"
      self.lines.append(cg)
      for child in token.src:
        self.generate_kernel(child)
      self.lines.append("}")  

    if token.arg == TokenType.DEFINE_ACC: 
      if token.ctx == ShapeOPS.MAX: tt = "-INFINITY"
      if token.ctx == ShapeOPS.SUM: tt = "0"
      cg = f"float {token.reg} = {tt};"
      self.loads[token.reg] = token
      self.lines.append(cg)
      
    elif token.arg == TokenType.LOOP:
      loop_var = token.reg
      start, end, step = token.ctx
      self.lines.append(f"for (int {loop_var} = {start}; {loop_var} < {end}; {loop_var} += {step}) {{")
      for child in token.src:
        self.generate_kernel(child)
      self.lines.append("}")  

    elif token.arg == TokenType.LOCAL:
      for child in token.src:
        expr = self.generate_kernel(child)
      cg = f"{token.reg if token.reg in self.loads else 'float ' + token.reg} = {expr};"
      self.loads[token.reg] = token
      self.lines.append(cg)

    elif token.arg == TokenType.IF:
      cond = ' && '.join([f"({self.generate_kernel(c)})" for c in token.cond])
      self.lines.append(f"if ( {cond} ) {{")
      for child in token.src: 
        self.generate_kernel(child) 
      self.lines.append("}")

    elif token.arg == TokenType.OP:
      arguments = []
      for child in token.src: 
        arguments.append(self.generate_kernel(child))
      cg = self.op_map[token.reg](*arguments)
      return cg

    elif token.arg == TokenType.INDEX: 
      expr = [] 
      for child in token.src: 
        expr.append(self.generate_kernel(child))
      cg = f"{expr[0]}[{expr[1]}]"
      return cg

    elif token.arg == TokenType.GLOBAL: 
      arguments = []
      for child in token.src: 
        arguments.append(self.generate_kernel(child))
      cg = f"{arguments[0]} = {arguments[1]};"
      self.lines.append(cg)

    return '\n'.join(self.lines)

