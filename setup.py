import numpy as np
from os import path
from Cython.Build import cythonize
from setuptools.extension import Extension
from setuptools import setup


ext = ".pyx"

defs = [("NPY_NO_DEPRECATED_API", 0)]
inc_path = np.get_include()
lib_path = path.join(inc_path, "..", "..", "random", "lib")

extensions = [
    Extension(
        "opaque.stats._stats",
        ["src/opaque/stats/_stats" + ext],
        include_dirs=[inc_path],
        library_dirs=[lib_path],
        libraries=["npyrandom"],
        define_macros=defs,
    ),
    Extension(
        "opaque.ood._tree_kernel",
        ["src/opaque/ood/_tree_kernel" + ext],
        include_dirs=[inc_path],
        define_macros=defs,
    ),
]


extensions = cythonize(
    extensions,
    compiler_directives={"language_level": 3},
)

setup(ext_modules=extensions)

