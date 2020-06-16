#!/usr/bin/env python

from setuptools import find_packages, setup

def dependencies_from_file(file_path):
    required = []
    with open(file_path) as f:
        for l in f.readlines():
            l_c = l.strip()
            # get not empty lines and ones that do not start with python
            # comment "#" (preceded by any number of white spaces)
            if l_c and not l_c.startswith('#'):
                required.append(l_c)
    return required

if __name__ == "__main__":
    setup(name='mlaas',
          description='Machine Learning as a Service',
          url='https://github.com/so-cool/mlaas',
          version='0.0.1',
          install_requires=dependencies_from_file('requirements.txt'),
          packages=find_packages(exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']),
          include_package_data=True)
