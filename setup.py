import numpy as np
from os import path
from Cython.Build import cythonize
from setuptools.extension import Extension
from setuptools import setup

ext = ".pyx"

inc_path = np.get_include()

extensions = [
    Extension(
        "opaque.stats._stats",
        ["src/opaque/stats/_stats" + ext],
        include_dirs=[inc_path],
    ),
]


extensions = cythonize(
    extensions,
    compiler_directives={"language_level": 3},
)

setup(ext_modules=extensions)

