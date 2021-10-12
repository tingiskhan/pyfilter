from setuptools import setup, find_packages
import os


NAME = "pyfilter"


def _get_version():
    folder = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(folder, f"{NAME}/__init__.py"), "r") as f:
        version_line = next(line for line in f.readlines() if line.strip().startswith("__version__"))
        version = version_line.split("=")[-1].strip().replace('"', "")

    return version.strip()


setup(
    name=NAME,
    version=_get_version(),
    author="Victor Gruselius",
    author_email="victor.gruselius@gmail.com",
    description="Package for performing Bayesian inference in state space models",
    packages=find_packages(),
    install_requires=["torch>1.5.0", "tqdm>=4.26", "numpy", "pytest"],
)
