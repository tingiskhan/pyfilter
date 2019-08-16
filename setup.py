from setuptools import setup, find_packages


setup(
    name='pyfilter',
    version='0.3.5',
    author='Victor Gruselius',
    author_email='victor.gruselius@gmail.com',
    description='Package for performing online Bayesian inference in state space models',
    packages=find_packages(),
    install_requires=[
        'matplotlib>=2.0.0',
        'pandas>=0.19.2',
        'scipy>=0.18.1',
        'torch>=1.0.0',
        'tqdm>=4.26',
        'scikit-learn>=0.20.1',
    ]
)
