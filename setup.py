from setuptools import setup

setup(
    name="Contextual",
    install_requires = ['DecoratorTools>=1.4'],
    version="0.7",
    packages = ['peak'],
    namespace_packages=['peak'],
    test_suite="test_context",
)
