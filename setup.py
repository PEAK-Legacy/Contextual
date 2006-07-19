from setuptools import setup, find_packages

setup(
    name="Contextual",
    install_requires = ['SymbolType'],
    version="0.5",
    packages = find_packages(),
    namespace_packages=['peak', 'peak.util'],
    test_suite="test_context",
)
