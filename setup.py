from setuptools import setup, find_packages

setup(
    name="stacierl",
    version="0.1dev",
    author="Lennart Karstensen",
    packages=find_packages(),
    install_requires=["torch", "numpy", "pyyaml"],
)
