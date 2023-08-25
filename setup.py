from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True
ext_modules = Extension("graver", sources=["graver.pyx"])
setup(
    ext_modules = cythonize(ext_modules),
    include_dirs = [np.get_include()]
    )
