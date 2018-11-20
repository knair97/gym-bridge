import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    sys.exit('Python < 3.5 is not supported')

setup(name='gym_bridge',
      version='0.0.1',
      install_requires=['gym', 'click', 'tqdm', 'pandas'],
      packages=find_packages()
      )
