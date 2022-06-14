import setuptools
from setuptools import setup

setup(
    name='sccala-sniip',
    version='0.1',
    author='Alexander Holas',
    description='Standardisation Framework for Core-Collapse Supernovae and Line Analysis Toolkit',
    url='http://github.com/AlexHls/Sccala',
    author_email='alexander.holas@h-its.org',
    license='GPLv2',
    project_urls={
        "Bug Tracker": "https://github.com/AlexHls/Sccala/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv2",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "sccala"},
    packages=setuptools.find_packages(where="sccala"),
    python_requires=">=3.6",
    scripts=[
        'bin/sccala-photometry',
        'bin/sccala-linefit',
        'bin/sccala-velocity-interpolation',
        'bin/sccala-photometry-interpolation',
    ],
)

