from setuptools import find_packages, setup

setup(
    name="context_gym",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'gym==0.23.1',
    ]
)