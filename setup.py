from setuptools import setup, find_packages
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy
import os
import platform

extra_compile_args = ['-O3', '-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':  # macOS
    brew_prefix = os.popen('brew --prefix').read().strip()
    libomp_path = f"{brew_prefix}/opt/libomp"
    
    extra_compile_args += ['-Xpreprocessor', '-fopenmp', f'-I{libomp_path}/include']
    extra_link_args += [f'-L{libomp_path}/lib', '-lomp']
else:
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']

ext_modules = [
    Extension(
        "balltree_setaxis.balltree",
        ["balltree_setaxis/balltree.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++"
    )
]

setup(
    name="balltree-setaxis",
    version="0.1.0",
    description="Ball tree with global PCA alignment for rotation-equivariant construction (based on balltree by Maksim Zhdanov)",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Coen van den Elsen",
    url="https://github.com/CoenvdE/balltree_setaxis",
    packages=find_packages(),
    # force: always regenerate balltree.cpp from balltree.pyx, so a stale checked-in .cpp is never compiled
    ext_modules=cythonize(ext_modules, force=True),
    install_requires=[
        "numpy",
        "torch",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    license="MIT",
)