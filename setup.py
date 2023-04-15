# -*- coding: utf-8 -*-
from codecs import open
import os
import re
from setuptools import setup

with open(os.path.join('yasklearn', '__init__.py'), 'r', encoding='utf8') as f:
    version = re.compile(r".*__version__ = '(.*?)'",
                         re.S).match(f.read()).group(1)

setup(name='yascikit-learn',
      packages=[
          'yasklearn', 'yasklearn.cluster', 'yasklearn.decomposition',
          'yasklearn.model_selection', 'yasklearn.ftrl_proximal'
      ],
      version=version,
      license='MIT License',
      description='Yet another scikit-learn',
      author='Yukino Ikegami',
      author_email='yknikgm@gmail.com',
      url='https://github.com/ikegami-yukino/yascikit-learn',
      keywords=['Machine Learning'],
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
      ],
      data_files=[('', ['README.md'])],
      long_description='%s' % (open('README.md', encoding='utf8').read()),
      long_description_content_type="text/markdown",
      install_requires=['numpy', 'scipy', 'pandas', 'scikit-learn', 'flati'])
