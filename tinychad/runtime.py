from __future__ import annotations
import os, ctypes, subprocess, tempfile
from typing import Union, Tuple, Optional, List, Dict
import numpy as np 

class ExecuteCProgram:
  def __init__(self, prg, bufs, fxn_name): 
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

  def run(self): 
    cfun = getattr(self.dll, self.fxn_name)
    cfun(*self.args)

