from setuptools import setup, find_packages

setup(
    name='mod2nn',
    version='0.1.0',
    license='proprietary',

    packages=find_packages(where='lib'),
    package_dir={'': 'lib'}
)