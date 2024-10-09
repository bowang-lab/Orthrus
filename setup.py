from setuptools import setup, find_packages

setup(
    name='orthrus',
    version='0.1.0',
    author='Philip Fradkin',
    author_email='phil.fradkin@gmail.com',
    description=(
        'Orthrus is a mature RNA model for RNA property prediction. '
        'It uses a mamba encoder backbone, a variant of '
        'state-space models specifically designed for long-sequence data, such as RNA.'
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/bowang-lab/Orthrus',
    packages=find_packages(),
    install_requires=[
        # Your dependencies
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
