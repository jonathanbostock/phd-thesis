# Setup our stuff

from setuptools import setup
from Cython.Build import cythonize

setup(name = "UpdateParticlesImport",
      ext_modules=cythonize("UpdateParticles.pyx"))
