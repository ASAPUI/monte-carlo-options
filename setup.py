"""
Setup script for Monte Carlo Options Pricing package
"""

from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="monte-carlo-options",
    version="1.0.0",
    author="Quantitative Finance Team",
    author_email="quant@example.com",
    description="Production-ready Monte Carlo options pricing system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/monte-carlo-options",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "web": [
            "streamlit>=1.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mco-pricer=cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)