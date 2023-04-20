from setuptools import setup, find_packages

setup(
    name="eve_rl",
    version="0.1",
    author="Lennart Karstensen",
    packages=find_packages(),
    install_requires=["torch", "numpy", "pyyaml"],
)
