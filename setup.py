from setuptools import find_packages, setup

setup(
    name="context_gym",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'gym==0.25.2',
        'Box2D==2.3.2',
        "box2d-py==2.3.8"
    ]
)