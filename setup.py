from setuptools import setup

setup(
    name="Contextual",
    install_requires = [
        'SymbolType>=1.0', 'DecoratorTools>=1.3', 'ProxyTypes>=0.9'
    ],
    version="0.7",
    packages = ['peak'],
    namespace_packages=['peak'],
    test_suite="test_context",
)
