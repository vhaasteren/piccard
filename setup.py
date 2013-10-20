import os
import sys
import piccard

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()


setup(
    name="piccard",
    version=emcee.__version__,
    author="Rutger van Haasteren",
    author_email="vhaasteren@gmail.com",
    packages=["piccard"],
    url="http://github.com/vhaasteren/piccard/",
    license="GPLv3",
    description="PTA analysis software",
    long_description=open("README.md").read() + "\n\n"
                    + "Changelog\n"
                    + "---------\n\n"
                    + open("HISTORY.rst").read(),
    package_data={"": ["LICENSE", "AUTHORS.md"]},
    include_package_data=True,
    install_requires=["numpy", "scipy"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
