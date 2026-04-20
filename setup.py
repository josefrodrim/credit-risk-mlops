from setuptools import find_packages, setup

setup(
    name="credit-risk-mlops",
    version="1.0.0",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.11",
)
