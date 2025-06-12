from setuptools import setup, find_packages

setup(
    name='tmw_project',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'POT',
        'python-lemon-binding',
    ],
    description='Temporal Masked Wasserstein (TMW) project',
    author='',
    author_email='',
)
