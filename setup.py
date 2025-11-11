"""
Setup script for breedrplus package
"""

from setuptools import setup, find_packages
import os

# Read the README file
readme_file = "README_PYTHON.md"
if os.path.exists(readme_file):
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Genomic Prediction Tools for Animal and Plant Breeding"

# Read requirements
requirements_file = "requirements.txt"
if os.path.exists(requirements_file):
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "bed-reader>=0.3.0",
        "xgboost>=1.5.0",
    ]

setup(
    name="breedrplus",
    version="0.0.0.9000",
    author="Fei Ge",
    author_email="higefee@gmail.com",
    description="Genomic Prediction Tools for Animal and Plant Breeding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/breedrplus",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "neural": ["tensorflow>=2.8.0"],
        "all": ["tensorflow>=2.8.0"],
    },
    include_package_data=True,
    zip_safe=False,
)

