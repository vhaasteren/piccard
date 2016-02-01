import os
import sys
import numpy

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

# Cython extensions
ext_modules=[
    Extension('piccard.jitterext',
             ['piccard/jitterext.pyx'],
             include_dirs = [numpy.get_include()],
             extra_compile_args=["-O2"]),
    Extension('piccard.choleskyext_omp',
             ['piccard/choleskyext_omp.pyx'],
             include_dirs = [numpy.get_include(), 'piccard/'],
             extra_link_args=["-liomp5"],
             extra_compile_args=["-O2", "-fopenmp", "-fno-wrapv"]),
    Extension('piccard.choleskyext',
             ['piccard/choleskyext.pyx'],
             include_dirs = [numpy.get_include(), 'piccard/'],
             extra_compile_args=["-O2", "-fno-wrapv"])  # 50% more efficient!
]


setup(
    name="piccard",
    version='2016.01',
    author="Rutger van Haasteren",
    author_email="vhaasteren@gmail.com",
    packages=["piccard"],
    url="http://github.com/vhaasteren/piccard/",
    license="GPLv3",
    description="PTA analysis software",
    long_description=open("README.md").read() + "\n\n"
                    + "Changelog\n"
                    + "---------\n\n"
                    + open("HISTORY.md").read(),
    package_data={"": ["README", "LICENSE", "AUTHORS.md"]},
    include_package_data=True,
    install_requires=["numpy", "scipy", "h5py", "healpy"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    ext_modules = cythonize(ext_modules)
)
