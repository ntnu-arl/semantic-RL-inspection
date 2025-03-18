from setuptools import find_packages
from distutils.core import setup

setup(
    name="semantic-RL-inspection",
    version="1.0.0",
    author="Grzegorz Malczyk",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="grzegorz.malczyk@ntnu.no",
    description="Semantic RL Inspection with Isaac Gym environments for Aerial Robots",
    install_requires=[
        "isaacgym",
        "aerial_gym",
        "matplotlib",
        "numpy",
        "torch",
        "pytorch3d",
        "warp-lang==1.0.0b5",
        "trimesh",
        "urdfpy",
        "numpy==1.23",
        "gymnasium",
        "rl-games",
        "sample-factory",
        "shimmy==0.2.1",
    ],
)
