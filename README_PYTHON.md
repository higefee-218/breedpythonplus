# breedrplus (Python Version)

Genomic Prediction Tools for Animal and Plant Breeding

## Description

`breedrplus` is a comprehensive Python package for genomic prediction in breeding programs. It provides tools for:

- **GBLUP**: Genomic Best Linear Unbiased Prediction with cross-validation
- **Bayesian Methods**: Simplified Bayesian regression (BayesA, BayesB, BayesC, BL)
- **Machine Learning**: Random Forest, XGBoost, Neural Networks (MLP, CNN), and PLS regression
- **Data Preparation**: PLINK file loading and genotype imputation

## Installation

### Prerequisites

- Python 3.7 or higher
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Package (Development)

```bash
# Clone or download the package
cd breedrplus

# Install in development mode
pip install -e .
```

## Dependencies

- `numpy>=1.20.0`: Numerical operations
- `pandas>=1.3.0`: Data manipulation
- `scipy>=1.7.0`: Scientific computing and linear algebra
- `scikit-learn>=1.0.0`: Machine learning models
- `bed-reader>=0.3.0`: PLINK .bed file reading
- `xgboost>=1.5.0`: Gradient boosting
- `tensorflow>=2.8.0`: Neural networks (MLP, CNN) - optional but recommended

## Quick Start

### 1. Load PLINK Data

```python
import pandas as pd
from breedrplus import load_plink

# Load PLINK files
data = load_plink(
    bed_file="genotypes.bed",
    bim_file="genotypes.bim",
    fam_file="genotypes.fam",
    pheno=phenotype_df,
    id_col_name="ID"
)
```

### 2. Run GBLUP

```python
from breedrplus import run_gblup, cv_gblup

# Fit GBLUP model
gblup_result = run_gblup(
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

### 3. Run Bayesian Methods

```python
from breedrplus import run_bglr, run_bglr_cv

# Fit Bayesian model
bayes_result = run_bglr(
    obj=data['snp_obj'],
    pheno=data['pheno'],
    trait_name="yield",
    id_col_name="ID",
    model_type="BayesA"
)

# Cross-validation
bayes_cv = run_bglr_cv(
    geno=genotype_matrix,
    pheno=data['pheno'],
    trait_name="yield",
    id_col_name="ID",
    model_type="BayesA",
    k_folds=5
)
```

### 4. Run Machine Learning Models

```python
from breedrplus import run_ml_model, run_ml_cv

# Train ML model
ml_result = run_ml_model(
    pheno=data['pheno'],
    geno=genotype_matrix,
    trait_name="yield",
    id_col_name="ID",
    model_type="RF"  # or "XGB", "MLP", "CNN", "PLS"
)

# Cross-validation
ml_cv = run_ml_cv(
    pheno=data['pheno'],
    geno=genotype_matrix,
    trait_name="yield",
    id_col_name="ID",
    model_type="RF",
    k=5
)
```

## Main Functions

### Data Preparation

- `load_plink()`: Load PLINK files into NumPy format
  - Parameters: `bed_file`, `bim_file`, `fam_file`, `pheno`, `id_col_name`, `impute_method`
  - Returns: Dictionary with `snp_obj` and `pheno`

### GBLUP

- `run_gblup()`: Fit GBLUP model and predict GEBVs
  - Parameters: `qc_results`, `trait_name`, `id_col_name`, `fixed_effects`, `drop_missing`, `predict_all`
  - Returns: Dictionary with `gebv_obs`, `variances`, `model_fit`, `G`, and optionally `gebv_all`

- `cv_gblup()`: Cross-validate GBLUP model
  - Parameters: `qc_results`, `trait_name`, `id_col_name`, `k`, `fixed_effects`, `seed`, `stratify_by`
  - Returns: Dictionary with `fold_results`, `overall`, and optionally `predictions`

### Bayesian Methods

- `run_bglr()`: Fit Bayesian genomic prediction model
  - Parameters: `obj`, `pheno`, `trait_name`, `id_col_name`, `model_type`, `n_iter`, `burn_in`
  - Returns: Dictionary with `gebv`, `snp_effects`, `residual_var`, `model_fit`
  - Note: This is a simplified implementation. For full MCMC-based Bayesian methods, consider using specialized packages.

- `run_bglr_cv()`: Cross-validate Bayesian model
  - Parameters: `geno`, `pheno`, `trait_name`, `id_col_name`, `model_type`, `n_iter`, `burn_in`, `k_folds`, `seed`
  - Returns: Dictionary with `y_true`, `y_pred`, `cv_correlation`, `cv_mse`

### Machine Learning

- `run_ml_model()`: Train machine learning model
  - Parameters: `pheno`, `geno`, `trait_name`, `id_col_name`, `model_type`, `**kwargs`
  - Model types: `"RF"` (Random Forest), `"XGB"` (XGBoost), `"MLP"` (Multi-layer Perceptron), `"CNN"` (Convolutional Neural Network), `"PLS"` (Partial Least Squares)
  - Returns: Dictionary with `gebv` and `model_fit`

- `run_ml_cv()`: Cross-validate machine learning model
  - Parameters: `pheno`, `geno`, `trait_name`, `id_col_name`, `model_type`, `k`, `seed`, `**kwargs`
  - Returns: Dictionary with `cv_gebv`, `cv_correlation`, `cv_mse`

## Example Workflow

A complete example workflow is available in `example_workflow.py`. To run it:

```bash
python example_workflow.py
```

Or in Python:

```python
# After installing the package
exec(open('example_workflow.py').read())
```

## Model Types

### GBLUP
- Uses genomic relationship matrix (G-matrix)
- Based on mixed model equations
- Provides variance components and heritability estimates

### Bayesian Methods
- **BayesA**: Each SNP has its own variance (approximated with ridge regression)
- **BayesB**: Mixture of point mass at zero and normal (approximated with elastic net)
- **BayesC**: Similar to BayesB with common variance (approximated with elastic net)
- **BL**: Bayesian Lasso with double exponential prior (approximated with Lasso)

**Note**: The Python implementation uses simplified approximations. For full MCMC-based Bayesian methods, the R package BGLR is recommended.

### Machine Learning
- **RF**: Random Forest (scikit-learn)
- **XGB**: XGBoost (gradient boosting)
- **MLP**: Multi-layer Perceptron (TensorFlow/Keras)
- **CNN**: Convolutional Neural Network (TensorFlow/Keras)
- **PLS**: Partial Least Squares Regression

## Differences from R Version

1. **Bayesian Methods**: The Python version uses simplified approximations (ridge regression, elastic net, lasso) instead of full MCMC sampling. For production use with Bayesian methods, consider using the R package BGLR or implementing proper MCMC sampling.

2. **GBLUP**: The Python version uses a simplified mixed model solver. For large datasets or complex models, the R package rrBLUP may be more efficient.

3. **PLINK Loading**: Uses `bed-reader` instead of `bigsnpr`. The data structure is slightly different but functionally equivalent.

4. **TensorFlow**: Neural network models (MLP, CNN) require TensorFlow. If TensorFlow is not available, these models will raise an error.

## Performance Notes

- For large datasets (>10,000 individuals), consider using `predict_all=False` in `run_gblup()` to save memory
- Machine learning models may require significant computational resources for large datasets
- GBLUP computation scales with O(n²) where n is the number of individuals
- Consider using parallel processing for cross-validation when possible

## License

GPL-3

## Author

Fei Ge (Python port)

## Citation

If you use this package, please cite the original R package and this Python implementation.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Known Issues

1. Bayesian methods are simplified approximations and may not match R BGLR results exactly
2. GBLUP mixed model solver is simplified and may have numerical stability issues with very large datasets
3. TensorFlow installation can be tricky on some systems (especially macOS with Apple Silicon)

## Future Improvements

- [ ] Full MCMC implementation for Bayesian methods
- [ ] More efficient GBLUP solver for large datasets
- [ ] GPU support for neural network models
- [ ] Additional machine learning models
- [ ] Better error handling and validation
- [ ] Unit tests
- [ ] Documentation with examples

