# fractal_framework/setup.py

from setuptools import setup, find_packages

setup(
    name="fractal_framework",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "networkx",
        "astunparse",
        # Include other dependencies as needed
    ],
    description="A fractal-inspired framework for code analysis and transformation",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
)
