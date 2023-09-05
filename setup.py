from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='tinychad',
    version='0.0.1', 
    packages=find_packages(),
    author='corranr',
    license='MIT',
    description='it will be small, but it will be based',
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url='http://github.com/tpat7187/tinychad',  
    install_requires=["numpy", "tqdm", "llvmlite"],
)