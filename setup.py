from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nama',
    version='0.1.0',
    packages=['nama'],
    install_requires=requirements,
    data_files=[('',['requirements.txt'])],
    url='https://github.com/bradhackinen/nama',
    description='A NAme MAtching tool' 
)