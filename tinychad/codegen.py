from __future__ import annotations
import ctypes, subprocess, tempfile
from typing import Union, Tuple, Optional, List, Dict
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS, LoadOPS, TokenType
from tinychad.helpers import DEBUG
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
  KERNEL_HEADER = "#import <math.h>"

  op_map = { 
    BinaryOPS.ADD: lambda tok1, tok2: f"{tok1} + {tok2}",
    BinaryOPS.SUB: lambda tok1, tok2: f"{tok1} - {tok2}",
    BinaryOPS.MUL: lambda tok1, tok2: f"{tok1} * {tok2}",
    BinaryOPS.DIV: lambda tok1, tok2: f"{tok1} / {tok2}",
    UnaryOPS.LOG: lambda tok1: f"log({tok1})",
    UnaryOPS.EXP: lambda tok1: f"exp({tok1})",
    UnaryOPS.SQRT: lambda tok1: f"sqrt({tok1})",
    UnaryOPS.RELU: lambda tok1: f"({tok1} > 0 ? {tok1} : 0)"
  }

  def __init__(self, tokens): 
    self.tokens = tokens
    self.kernel = self.generate_kernel(tokens)

  def generate_kernel(self, toks):
    lines = [] 
    lines.append(self.KERNEL_HEADER)
    for tok in toks:
      if tok.type == TokenType.FUNCSTART: 
        cg = f"void {tok.args[0]}({', '.join(['float* ' + _ for _ in tok.args[1]])}) {{"
        tok.codegen = cg 
        lines.append(cg)

      elif tok.type == TokenType.FUNCEND: lines.append("}")

      elif tok.type == TokenType.LOOPSTOP: lines.append("}")

      elif tok.type == TokenType.LOOPSTART:
        cg = f"for (int {tok.reg}={tok.start}; {tok.reg}<{tok.iters}; {tok.reg}+={tok.inc}) {{"
        tok.codegen = cg
        lines.append(cg)

      # load from buffer
      elif tok.type == TokenType.LOAD: 
        cg = f"float {tok.reg} = {tok.args[0]}[{tok.args[1]}];"
        tok.codegen = cg 
        lines.append(cg)

      # stores into buffer
      elif tok.type == TokenType.GLOBAL: 
        cg = f"{tok.reg}[{tok.args[1].reg}] = {tok.args[0].reg};"
        tok.codegen = cg
        lines.append(cg)
      
      # stores into float
      elif tok.type == TokenType.LOCAL: 
        if tok.args.type == TokenType.OP: 
          op_string = self.codegen_operation(tok.args) 
          cg = f"float {tok.reg}={op_string};"
          tok.codegen = cg 
          lines.append(cg)

      elif tok.type == TokenType.DEFINE_ACC: 
        cg = f"float {tok.reg}=0;"
        tok.codegen = cg
        lines.append(cg)

      elif tok.type == TokenType.ACC: 
        # I dont really like this, its too hacky
        cg = f"{tok.args[1].reg} += {tok.args[0].args[0]}[{tok.args[0].args[1].reg}];"
        lines.append(cg)

    kern = '\n'.join(lines)
    if DEBUG: print(kern)
    return kern

  def codegen_operation(self, token):
    args = [f'''{x.args[0]}[{x.args[1]}]''' for x in token.args[1]]
    cg = self.op_map[token.args[0]](*args)
    return cg











