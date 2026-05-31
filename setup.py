# from setuptools import setup

# setup(name='gym_rescue',
#       version='1.0.0',
#       install_requires=['gym==0.26.0', 'matplotlib', 'numpy>=1.21.6', 'unrealcv>=1.1.5', 'wget', 'opencv-python','docker'],  # And any other dependencies foo needs
# )
from setuptools import setup, find_packages

setup(
    name="gym_rescue",
    version="1.0.0",
    packages=find_packages(include=["gym_rescue", "gym_rescue.*"]),
    install_requires=[
        "gym==0.26.0",
        "matplotlib",
        "numpy>=1.21.6",
        "unrealcv>=1.1.5",
        "wget",
        "opencv-python",
        "docker",
    ],
)