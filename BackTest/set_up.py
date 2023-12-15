from setuptools import setup, find_packages

setup(
    name='BackTest',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'feather-format'
    ]
)