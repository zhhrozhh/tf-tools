import sys
from os import path,listdir
from setuptools import setup,find_packages
__version__ = "0.0.4.8"
REQUIRES = [
    'tensorflow>=1.3.0',
    'numpy>=1.12.1',
    'matplotlib>=2.0.2',
    'Keras>=2.1.3'
]
setup(
    name = 'tf-tools',
    author = 'zhhrozhh',
    author_email = 'zhangh40@msu.edu',
    url = 'https://github.com/zhhrozhh/tf-tools',
    version = __version__,
    license = 'MIT',
    classifiers = [
        'Programming Language :: Python :: 3.5'
    ],
    keywords = 'tensorflow tools',
    packages = find_packages(),
    install_requires = REQUIRES
)
