import numpy as np
from Cython.Build import cythonize
from setuptools.extension import Extension
from setuptools import setup


extensions = [
    Extension(
        "opaque.stats._stats",
        ["src/opaque/stats/_stats.pyx"],
        include_dirs=[np.get_include()],
    ),
]


extensions = cythonize(
    extensions,
    compiler_directives={"language_level": 3},
    gdb_debug=True,
)

setup(ext_modules=extensions)

