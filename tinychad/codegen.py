from __future__ import annotations
import os, ctypes, subprocess, tempfile
from typing import Union, Tuple, Optional, List, Dict
from tinychad.ops_type import UnaryOPS, BinaryOPS, ShapeOPS, ReshapeOPS, LoadOPS, TokenType, DEBUG
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


class CPrinter:  
  @classmethod 
  def generate_kernel(self, toks: Tokenizer):
    lines = [] 
    for tok in  toks.token_stream:
      if tok.type == TokenType.FUNCSTART: 
        cg = f"void {tok.args[0]}({', '.join(['float* ' + _ for _ in tok.args[1]])}) {{"
        tok.codegen = cg 
        lines.append(cg)

      elif tok.type == TokenType.FUNCEND: lines.append("}")

      elif tok.type == TokenType.LOOPSTOP: lines.append("}")

      elif tok.type == TokenType.LOOPSTART:
        cg = f"for (int {tok.args[-1]}={tok.start}; {tok.args[-1]}<{tok.iters}; {tok.args[-1]}+={tok.inc}) {{"
        tok.codegen = cg
        lines.append(cg)

      elif tok.type == TokenType.LOAD: 
        cg = f"float {tok.reg} = {tok.args[0]}[{tok.args[1]}];"
        tok.codegen = cg 
        lines.append(cg)

      elif tok.type == TokenType.OP: 
        if tok.args[0] in BinaryOPS:
          op_token = ops_to_toks[tok.args[0]]
          cg = f"float {tok.reg} = {f' {op_token} '.join([_.reg for _ in tok.args[1]])};"
          tok.codegen = cg 
          lines.append(cg)
        else: 
          op_token =  ops_to_toks[tok.args[0]]
          outreg = tok.args[1][0].reg
          cg = f"float {tok.reg} = {op_token}({outreg});"
          lines.append(cg)
          tok.codegen = cg 

      elif tok.type == TokenType.GLOBAL: 
        cg = f"{tok.args[0]}[{tok.args[1]}] = {tok.reg};"
        tok.codegen = cg 
        lines.append(cg)

    kern = '\n'.join(lines)
    if DEBUG: print(kern)
     
    return kern

ops_to_toks = { 
  BinaryOPS.ADD: '+',
  BinaryOPS.SUB: '-',
  BinaryOPS.MUL: '*',
  BinaryOPS.DIV: '/',
  UnaryOPS.RELU: 'relu'
}
