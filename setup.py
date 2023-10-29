from pkg_resources import parse_requirements
from setuptools import setup

with open("requirements.txt") as f:
    requirements = [str(req) for req in parse_requirements(f)]

setup(
    name="vit",
    version="0.1.0",
    py_modules=["vit"],
    python_requires=">=3.11",
    install_requires=requirements,
)
