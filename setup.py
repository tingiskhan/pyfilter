from setuptools import setup, find_packages
import os
import sys

from pip._internal.network.session import PipSession
from pip._internal.req import parse_requirements

NAME = "pyfilter"

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    read_license = f.read()

NAME = "pyfilter"


def requirements_from_file(filename):
    """Parses a pip requirements file into a list."""
    requirements = parse_requirements(filename, session=PipSession())
    return [str(requirement.requirement) for requirement in requirements]


def _get_version():
    folder = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(folder, f"{NAME}/__init__.py"), "r") as f:
        version_line = next(line for line in f.readlines() if line.strip().startswith("__version__"))
        version = version_line.split("=")[-1].strip().replace('"', "")

    return version.strip()


# This grabs the requirements from *requirements.in
requirements_in = requirements_from_file('requirements.in')
setup_requires_in = requirements_from_file('setup_requirements.in')
tests_require_in = requirements_from_file('tests_requirements.in')

# See https://pytest-runner.readthedocs.io/en/latest/#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

setup_requires = setup_requires_in + pytest_runner,

extras_require = {
    'setup': setup_requires,
    'test': tests_require_in,
}

setup(
    name=NAME,
    version=_get_version(),
    author="Victor Gruselius",
    author_email="victor.gruselius@gmail.com",
    description="Package for performing Bayesian inference in state space models",
    packages=find_packages(),
    install_requires=["torch>1.5.0", "tqdm>=4.26", "numpy", "pytest"],
)
