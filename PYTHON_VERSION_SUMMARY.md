# Python Version of breedRplus - Summary

## Overview

This Python version of the breedRplus R package provides genomic prediction tools for animal and plant breeding. The package has been ported from R to Python with equivalent functionality.

## Package Structure

```
breedrplus/
├── __init__.py          # Package initialization and exports
├── data_prep.py         # PLINK file loading and data preparation
├── gblup_utils.py       # GBLUP (Genomic Best Linear Unbiased Prediction)
├── bayes_utils.py       # Bayesian genomic prediction methods
└── ml.py                # Machine learning models (RF, XGBoost, MLP, CNN, PLS)

setup.py                 # Package installation script
requirements.txt         # Python dependencies
example_workflow.py      # Example usage script
README_PYTHON.md         # Python version documentation
```

## Key Functions

### Data Preparation
- `load_plink()`: Load PLINK files (.bed, .bim, .fam) into NumPy arrays
  - Supports genotype imputation (mode, mean, zero)
  - Aligns phenotype data with genotype data

### GBLUP
- `run_gblup()`: Fit GBLUP model and predict GEBVs
  - Computes genomic relationship matrix (G-matrix)
  - Estimates variance components
  - Predicts breeding values for all individuals
- `cv_gblup()`: K-fold cross-validation for GBLUP
  - Supports stratification
  - Returns fold-level and overall metrics

### Bayesian Methods
- `run_bglr()`: Fit Bayesian genomic prediction model
  - Models: BayesA, BayesB, BayesC, BL (Bayesian Lasso)
  - **Note**: Uses simplified approximations (ridge regression, elastic net, lasso)
  - For full MCMC-based Bayesian methods, use the R package BGLR
- `run_bglr_cv()`: Cross-validation for Bayesian models

### Machine Learning
- `run_ml_model()`: Train ML model for genomic prediction
  - Models: RF (Random Forest), XGB (XGBoost), MLP (Neural Network), CNN (Convolutional Neural Network), PLS (Partial Least Squares)
- `run_ml_cv()`: Cross-validation for ML models

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package (development mode)
pip install -e .
```

## Dependencies

- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **scipy**: Scientific computing and linear algebra
- **scikit-learn**: Machine learning models
- **bed-reader**: PLINK .bed file reading
- **xgboost**: Gradient boosting
- **tensorflow**: Neural networks (optional, for MLP and CNN)
- **openpyxl**: Excel file reading (for example data)

## Key Differences from R Version

1. **Bayesian Methods**: 
   - Python version uses simplified approximations (ridge regression, elastic net, lasso)
   - R version uses full MCMC sampling via BGLR package
   - For production use with Bayesian methods, consider using the R package

2. **GBLUP**:
   - Python version uses a simplified mixed model solver
   - R version uses rrBLUP::mixed.solve which may be more efficient for large datasets
   - Results should be similar but may differ slightly due to numerical methods

3. **PLINK Loading**:
   - Python version uses `bed-reader` package
   - R version uses `bigsnpr` package
   - Data structures are different but functionally equivalent

4. **Neural Networks**:
   - Python version uses TensorFlow/Keras
   - R version uses Keras (R interface to TensorFlow)
   - Functionally equivalent

## Usage Example

```python
from breedrplus import load_plink, run_gblup, cv_gblup

# Load data
data = load_plink(
    bed_file="genotypes.bed",
    bim_file="genotypes.bim",
    fam_file="genotypes.fam",
    pheno=phenotype_df,
    id_col_name="ID"
)

# Run GBLUP
result = run_gblup(
    qc_results=data,
    trait_name="yield",
    id_col_name="ID",
    predict_all=True
)

# Cross-validation
cv_result = cv_gblup(
    qc_results=data,
    trait_name="yield",
    id_col_name="ID",
    k=5
)
```

## Performance Notes

- For large datasets (>10,000 individuals), GBLUP computation scales as O(n²)
- Consider using `predict_all=False` to save memory
- Machine learning models may require significant computational resources
- TensorFlow is required for MLP and CNN models

## Limitations

1. **Bayesian Methods**: Simplified approximations may not match R BGLR results exactly
2. **GBLUP Solver**: Simplified implementation may have numerical stability issues with very large datasets
3. **TensorFlow**: Installation can be tricky on some systems (especially macOS with Apple Silicon)

## Future Improvements

- [ ] Full MCMC implementation for Bayesian methods
- [ ] More efficient GBLUP solver for large datasets
- [ ] GPU support for neural network models
- [ ] Additional machine learning models
- [ ] Better error handling and validation
- [ ] Unit tests
- [ ] More comprehensive documentation with examples

## Testing

To test the package:

```bash
# Run example workflow
python example_workflow.py
```

Note: The example workflow may need adjustment of file paths to match your data location.

## License

GPL-3 (same as R version)

## Author

Fei Ge (Python port)

## Citation

If you use this package, please cite the original R package and this Python implementation.

