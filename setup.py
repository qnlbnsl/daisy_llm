from setuptools import setup, find_packages

setup(
    name='mypackage',
    version='1.0.0',
    description='My Python package',
    author='John Doe',
    author_email='john.doe@example.com',
    url='https://github.com/username/mypackage',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ],
)
