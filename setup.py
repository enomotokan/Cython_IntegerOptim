from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True
setup(
    ext_modules = cythonize("graver.pyx"),
    include_dirs = [np.get_include()]
    )
