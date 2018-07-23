from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
  name = "krr",
  cmdclass = {"build_ext": build_ext},
  ext_modules =
  [
    Extension("krr",
              ["krr.pyx"],
              extra_compile_args = ["-O3", "-fopenmp", "-ffast-math", "-march=native"],
              extra_link_args=['-fopenmp'],
              include_dirs = [numpy.get_include()]
              )
  ]
)

