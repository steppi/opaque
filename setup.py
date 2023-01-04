import numpy as np
from os import path
from Cython.Build import cythonize
from setuptools.extension import Extension
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

ext = ".pyx"

defs = [("NPY_NO_DEPRECATED_API", 0)]
inc_path = np.get_include()
lib_path = path.join(inc_path, "..", "..", "random", "lib")

extensions = [
    Extension(
        "opaque.stats._stats",
        ["opaque/stats/_stats" + ext],
        include_dirs=[inc_path],
        library_dirs=[lib_path],
        libraries=["npyrandom"],
        define_macros=defs,
    ),
    Extension(
        "opaque.ood._tree_kernel",
        ["opaque/ood/_tree_kernel" + ext],
        include_dirs=[inc_path],
        define_macros=defs,
    ),
]


extensions = cythonize(
    extensions,
    compiler_directives={"language_level": 3},
)

setup(
    name="opaque",
    version="0.0.0",
    description="Outlier prevalence analysis for quantification"
    " of unknown entities.",
    author="opaque developers, Harvard Medical School",
    author_email="albert.steppi@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    install_requires=[
        "appdirs",
        "liblinear-official",
        "cython",
        "numba",
        "pymc",
        "scikit-learn",
        "statsmodels",
    ],
    extras_require={"test": ["mpmath", "pandas", "pytest", "pytest-cov"]},
    ext_modules=extensions,
    include_package_data=True,
)
