"""
Setup for the hermitedpp package.
"""
from setuptools import Extension, find_packages, setup

from Cython.Build import cythonize


short_description = "Implementation of Hermite-Gaussian determinantal point processes."

with open("README.md", encoding="utf-8") as rf:
    long_description = "\n" + rf.read()



ext_modules = [
    Extension(
        name="semi_circle_cython_backend",
        sources=["hermitedpp/semi_circle_cython_backend.pyx"]
    )
]

setup(
    name="hermitedpp",
    version="0.1.0",
    description=short_description,
    long_description=long_description,
    author="nicholas92457",
    author_email="nicholas92457@gchq.gov.uk",
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    licence="OGL",
    url="https://bitbucket.tdx.oneit.gov.uk/projects/RML/repos/hermitedpp",
    packages=find_packages(),
    install_requires=["dppy", "numpy", "scipy"],
    setup_requires=["setuptools>=18.0", "cython>=0.29"],
    ext_modules=cythonize(ext_modules)
)
