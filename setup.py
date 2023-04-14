from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nama',
    version='1.0.0',
    packages=['nama'],
    install_requires=requirements,
    url='https://github.com/bradhackinen/nama',
    description='A NAme MAtching tool' 
)