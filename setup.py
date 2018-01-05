from setuptools import setup
from setuptools import find_packages

setup(name='multiscalegnn',
      version='0.1',
      description='Multi scale graph neural networks in PyTorch',
      author='Arsene Fansi',
      author_email='arsene.fansi@gmail.com',
      url='https://afansi.github.io',
      download_url='https://github.com/afansi/multiscalegnn',
      license='MIT',
      install_requires=['numpy',
                        'torch',
                        'scipy'
                        ],
      package_data={'multiscalegnn': ['README.md']},
      packages=find_packages())
