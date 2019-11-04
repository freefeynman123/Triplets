from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='triplets',
    version='0.1',
    author='Lukasz Bala',
    description='Triplet networks',
    long_description=open('README.md').read(),
    url='https://github.com/freefeynman123/triplets',
    install_requires=requirements,
    license='MIT',
    packages=['triplets'],
    zip_safe=True
)