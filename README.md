# breedrplus (Python Version)

Genomic Prediction Tools for Animal and Plant Breeding

## Description

`breedrplus` is a comprehensive Python package for genomic prediction in breeding programs. It provides tools for:

- **GBLUP**: Genomic Best Linear Unbiased Prediction with cross-validation
- **Bayesian Methods**: Simplified Bayesian regression (BayesA, BayesB, BayesC, BL)
- **Machine Learning**: Random Forest, XGBoost, Neural Networks (MLP, CNN), and PLS regression
- **Data Preparation**: PLINK file loading and genotype imputation

**Note**: We **strongly recommend using Python for machine learning methods** (especially neural networks) due to easier TensorFlow installation, better performance, and native implementation. See `ML_PYTHON_GUIDE.md` for a complete ML guide.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip

### Install Dependencies

```bash
# Install all dependencies
pip install numpy pandas scipy scikit-learn bed-reader xgboost tensorflow

# Or install from requirements.txt
pip install -r requirements.txt
```

**Note for Neural Networks**: TensorFlow is required for MLP and CNN models. If you encounter installation issues:

```bash
# For CPU-only version (lighter, no GPU)
pip install tensorflow-cpu

# Or install specific version
pip install tensorflow==2.13.0
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

# Extract genotype matrix (if needed for other methods)
geno_matrix = data['snp_obj']['genotypes']

# Fit GBLUP model
gblup_result = run_gblup(
    qc_results=data,
    trait_name="yield",
    id_col_name="ID",
    fixed_effects=None,  # None = intercept only, or ["sex", "age"] for fixed effects
    predict_all=True
)

# View results
print("GEBVs for observed individuals:")
print(gblup_result['gebv_obs'].head())

# View variance components and heritability
print("\nVariance components:")
print(gblup_result['variances'])

# Cross-validation
cv_result = cv_gblup(
    qc_results=data,
    trait_name="yield",
    id_col_name="ID",
    k=5,
    fixed_effects=None,  # None = intercept only
    seed=2025
)

# View CV results
print("\nOverall CV metrics:")
print(cv_result['overall'])
print("\nPer-fold results:")
print(cv_result['fold_results'])
```

### 3. Run Bayesian Methods

```python
from breedrplus import run_bglr, run_bglr_cv

# Extract genotype matrix (required for Bayesian CV)
geno_matrix = data['snp_obj']['genotypes']

# Fit Bayesian model
bayes_result = run_bglr(
    obj=data['snp_obj'],
    pheno=data['pheno'],
    trait_name="yield",
    id_col_name="ID",
    model_type="BayesA",  # Options: "BayesA", "BayesB", "BayesC", "BL"
    n_iter=5000,
    burn_in=1000
)

# View results
print("GEBVs:")
print(bayes_result['gebv'].head())
print("\nSNP Effects (first 10):")
print(bayes_result['snp_effects'].head(10))

# Cross-validation
bayes_cv = run_bglr_cv(
    geno=geno_matrix,
    pheno=data['pheno'],
    trait_name="yield",
    id_col_name="ID",
    model_type="BayesA",
    n_iter=5000,
    burn_in=1000,
    k_folds=5,
    seed=123
)

# View CV metrics
print(f"\nCV Correlation: {bayes_cv['cv_correlation']:.4f}")
print(f"CV MSE: {bayes_cv['cv_mse']:.4f}")
```

### 4. Run Machine Learning Models

**Python is recommended for ML methods** - See `ML_PYTHON_GUIDE.md` for complete guide.

```python
from breedrplus import run_ml_model, run_ml_cv
import matplotlib.pyplot as plt

# Extract genotype matrix (required for ML functions)
geno_matrix = data['snp_obj']['genotypes']

# Train Random Forest model
rf_result = run_ml_model(
    pheno=data['pheno'],
    geno=geno_matrix,
    trait_name="yield",
    id_col_name="ID",
    model_type="RF",  # Options: "RF", "XGB", "MLP", "CNN", "PLS"
    n_trees=500       # RF-specific parameter
)

# View predictions
print("RF Predictions:")
print(rf_result['gebv'].head())

# Cross-validation
rf_cv = run_ml_cv(
    pheno=data['pheno'],
    geno=geno_matrix,
    trait_name="yield",
    id_col_name="ID",
    model_type="RF",
    k=5,
    seed=123,
    n_trees=500
)

# View accuracy metrics
print(f"\nCV Correlation: {rf_cv['cv_correlation']:.4f}")
print(f"CV MSE: {rf_cv['cv_mse']:.4f}")

# Train Neural Network (MLP) - Much easier in Python!
mlp_result = run_ml_model(
    pheno=data['pheno'],
    geno=geno_matrix,
    trait_name="yield",
    id_col_name="ID",
    model_type="MLP",
    epochs=50
)

# Cross-validation for MLP
mlp_cv = run_ml_cv(
    pheno=data['pheno'],
    geno=geno_matrix,
    trait_name="yield",
    id_col_name="ID",
    model_type="MLP",
    k=5,
    seed=123,
    epochs=50
)

# Plot predictions vs observed
plt.figure(figsize=(8, 6))
y_true = data['pheno']['yield'].values
y_pred = mlp_cv['cv_gebv']['gebv'].values
plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
         'r--', lw=2, label='1:1 line')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('MLP Cross-Validation Predictions')
plt.legend()
plt.savefig('mlp_cv_predictions.png', dpi=100)
plt.close()
```

## Main Functions

### Data Preparation

- `load_plink()`: Load PLINK files into NumPy format
  - Parameters: `bed_file`, `bim_file`, `fam_file`, `pheno`, `id_col_name`, `impute_method`
  - Returns: Dictionary with `snp_obj` and `pheno`

### GBLUP

- `run_gblup()`: Fit GBLUP model and predict GEBVs
  - Parameters: `qc_results`, `trait_name`, `id_col_name`, `fixed_effects`, `drop_missing`, `predict_all`
  - `fixed_effects`: List of column names (e.g., `["sex", "age"]`) or `None` for intercept only
    - **Note**: R version uses formula syntax (`~1`, `~sex+age`), Python uses list of column names
  - Returns: Dictionary with `gebv_obs`, `variances` (sigma_g, sigma_e, h2), `model_fit`, `G`, and optionally `gebv_all`

- `cv_gblup()`: Cross-validate GBLUP model
  - Parameters: `qc_results`, `trait_name`, `id_col_name`, `k`, `fixed_effects`, `seed`, `stratify_by`, `drop_missing`, `return_fold_preds`
  - `fixed_effects`: List of column names or `None` for intercept only
  - Returns: Dictionary with `fold_results` (DataFrame), `overall` (Series with mean metrics), and optionally `predictions` (DataFrame)

### Bayesian Methods

- `run_bglr()`: Fit Bayesian genomic prediction model
  - Parameters: `obj`, `pheno`, `trait_name`, `id_col_name`, `model_type`, `n_iter`, `burn_in`
  - Returns: Dictionary with `gebv`, `snp_effects`, `residual_var`, `model_fit`
  - Note: This is a simplified implementation. For full MCMC-based Bayesian methods, consider using specialized packages.

- `run_bglr_cv()`: Cross-validate Bayesian model
  - Parameters: `geno`, `pheno`, `trait_name`, `id_col_name`, `model_type`, `n_iter`, `burn_in`, `k_folds`, `seed`
  - Returns: Dictionary with `y_true`, `y_pred`, `cv_correlation`, `cv_mse`

### Machine Learning

**See `ML_PYTHON_GUIDE.md` for complete ML guide and examples.**

- `run_ml_model()`: Train machine learning model
  - Parameters: `pheno`, `geno`, `trait_name`, `id_col_name`, `model_type`, `**kwargs`
  - Model types: 
    - `"RF"` (Random Forest) - scikit-learn, parameter: `n_trees` (default: 500)
    - `"XGB"` (XGBoost) - xgboost, parameter: `n_rounds` (default: 100)
    - `"MLP"` (Multi-layer Perceptron) - TensorFlow/Keras, parameter: `epochs` (default: 50)
    - `"CNN"` (Convolutional Neural Network) - TensorFlow/Keras, parameter: `epochs` (default: 50)
    - `"PLS"` (Partial Least Squares) - scikit-learn, parameter: `n_comp` (default: min(50, n_features))
  - Returns: Dictionary with `gebv` (DataFrame) and `model_fit`

- `run_ml_cv()`: Cross-validate machine learning model
  - Parameters: `pheno`, `geno`, `trait_name`, `id_col_name`, `model_type`, `k`, `seed`, `**kwargs`
  - Returns: Dictionary with `cv_gebv` (DataFrame), `cv_correlation` (float), `cv_mse` (float)

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

**For Machine Learning examples**, see `ML_PYTHON_GUIDE.md` which includes:
- Complete ML workflow
- Model comparison examples
- Visualization examples
- Troubleshooting guide

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

2. **GBLUP**: 
   - The Python version uses a simplified mixed model solver. For large datasets or complex models, the R package rrBLUP may be more efficient.
   - **`fixed_effects` parameter**: R uses formula syntax (`~1`, `~sex+age`), Python uses a list of column names (`None` or `["sex", "age"]`). This is documented in function docstrings.

3. **PLINK Loading**: Uses `bed-reader` instead of `bigsnpr`. The data structure is slightly different but functionally equivalent.

4. **TensorFlow**: Neural network models (MLP, CNN) require TensorFlow. Installation is straightforward: `pip install tensorflow`. This is much easier than R's keras/TensorFlow setup.

5. **Machine Learning**: Python is **recommended** for ML methods due to:
   - Easier TensorFlow installation
   - Better performance (native implementation)
   - More active development
   - Better production deployment options

**See `PYTHON_R_COMPARISON.md` for detailed comparison of implementations.**

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
3. TensorFlow installation: Generally straightforward with `pip install tensorflow`. For Apple Silicon (M1/M2), may need `pip install tensorflow-macos`

## Additional Resources

- **`ML_PYTHON_GUIDE.md`**: Complete guide for machine learning methods
- **`PYTHON_R_COMPARISON.md`**: Detailed comparison between Python and R implementations
- **`README.md`**: Main package README with R examples
- **`USER_MANUAL.md`**: Comprehensive user manual

## Future Improvements

- [ ] Full MCMC implementation for Bayesian methods
- [ ] More efficient GBLUP solver for large datasets
- [ ] GPU support for neural network models
- [ ] Additional machine learning models
- [ ] Better error handling and validation
- [ ] Unit tests
- [ ] Documentation with examples

