from setuptools import setup, find_packages
from pyfilter import __version__

setup(
    name='pyfilter',
    version=__version__,
    author='Victor Gruselius',
    author_email='victor.gruselius@gmail.com',
    description='Package for performing online Bayesian inference in state space models',
    packages=find_packages(),
    install_requires=[
        'scipy>=0.18.1',
        'torch>=1.0.0',
        'tqdm>=4.26',
        'numpy>=1.16.4'
    ]
)
