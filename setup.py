#!/usr/bin/env python
"""Distutils setup file"""
import ez_setup
ez_setup.use_setuptools()
from setuptools import setup

# Metadata
PACKAGE_NAME = "Contextual"
PACKAGE_VERSION = "0.7a1"
PACKAGES = ['peak']
def get_description():
    # Get our long description from the documentation
    f = file('README.txt')
    lines = []
    for line in f:
        if not line.strip():
            break     # skip to first blank line
    for line in f:
        if line.startswith('.. contents::'):
            break     # read to table of contents
        lines.append(line)
    f.close()
    return ''.join(lines)

setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    url = "http://pypi.python.org/pypi/Contextual",
    download_url = "http://peak.telecommunity.com/snapshots/",
    description='Replace globals with context-safe variables and services',
    long_description = get_description(),
    author="Phillip J. Eby",
    author_email="peak@eby-sarna.com",
    license="PSF or ZPL",
    test_suite = 'test_context',
    packages = PACKAGES,
    namespace_packages = PACKAGES,
    install_requires = ['DecoratorTools>=1.6']
)

