"""
Setup for the cython backend only.
"""
from setuptools import Extension, setup

from Cython.Build import cythonize
from Cython.Distutils import build_ext


ext_modules = [
    Extension(
        name="semi_circle_cython_backend",
        sources=["hermitedpp/semi_circle_cython_backend.pyx"]
    )
]

setup(
    name="semi_circle_cython_backend",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(ext_modules)
)
