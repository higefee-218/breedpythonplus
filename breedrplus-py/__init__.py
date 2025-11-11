"""
breedrplus - Genomic Prediction Tools for Animal and Plant Breeding

A comprehensive Python package for genomic prediction in breeding programs.
Provides tools for GBLUP, Bayesian methods, and machine learning approaches
for genomic selection.
"""

from .data_prep import load_plink
from .gblup_utils import run_gblup, cv_gblup
from .bayes_utils import run_bglr, run_bglr_cv
from .ml import run_ml_model, run_ml_cv

__version__ = "0.0.0.9000"
__author__ = "Fei Ge"

__all__ = [
    "load_plink",
    "run_gblup",
    "cv_gblup",
    "run_bglr",
    "run_bglr_cv",
    "run_ml_model",
    "run_ml_cv",
]

