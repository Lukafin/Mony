#!/usr/bin/env python
"""Setup script for Mony CLI tool."""

from setuptools import setup, find_packages

setup(
    name="mony",
    version="0.1.0",
    description="Generate UI concept images by combining project briefs with designer personas and the OpenRouter image API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/Lukafin/mony",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mony=mony.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="ui design image-generation openrouter cli",
)
