# R to Python Package Mapping

This document shows which Python packages/libraries were used to replace the R packages in the breedRplus Python version.

## R Package → Python Package Mapping

### 1. **bigsnpr** (R) → **bed-reader** (Python)

**R Package**: `bigsnpr`
- Used for: Reading and handling large PLINK files (.bed, .bim, .fam)
- Functions: `snp_readBed()`, `snp_attach()`, `snp_fastImputeSimple()`, `bed()`, `bed_tcrossprodSelf()`, `bed_scaleBinom()`

**Python Replacement**: `bed-reader`
- Package: `bed-reader` (pip install bed-reader)
- Used for: Reading PLINK .bed files efficiently
- Functions: `open_bed()`, `.read()`
- Note: Genomic relationship matrix computation is done manually using NumPy

**Why**: `bed-reader` is a Python-native package specifically designed for reading PLINK binary files efficiently, similar to how `bigsnpr` handles them in R.

---

### 2. **rrBLUP** (R) → **scipy + custom implementation** (Python)

**R Package**: `rrBLUP`
- Used for: GBLUP (Genomic Best Linear Unbiased Prediction)
- Main function: `mixed.solve()` - solves mixed model equations

**Python Replacement**: **scipy.linalg.solve + custom mixed model solver**
- Package: `scipy` (specifically `scipy.linalg.solve`, `scipy.linalg.cholesky`)
- Used for: Solving linear systems and mixed model equations
- Implementation: Custom iterative solver for variance component estimation

**Why**: There's no direct Python equivalent to `rrBLUP::mixed.solve()`. The Python version implements a simplified mixed model solver using:
- `scipy.linalg.solve()` for solving linear systems
- Iterative method of moments for variance component estimation
- Henderson's mixed model equations

**Limitation**: The Python implementation is simplified and may not match R results exactly for complex cases. For production use with large datasets, the R package `rrBLUP` may be more efficient.

---

### 3. **BGLR** (R) → **scikit-learn** (Python) - Simplified

**R Package**: `BGLR`
- Used for: Bayesian Genomic Regression
- Models: BayesA, BayesB, BayesC, BL (Bayesian Lasso)
- Method: Full MCMC (Markov Chain Monte Carlo) sampling

**Python Replacement**: **scikit-learn** (simplified approximations)
- Package: `scikit-learn`
- Used for:
  - **BayesA**: `Ridge` regression (ridge regression with L2 penalty)
  - **BayesB**: `ElasticNet` (elastic net with L1+L2 penalties)
  - **BayesC**: `ElasticNet` with different parameters
  - **BL (Bayesian Lasso)**: `Lasso` regression (L1 penalty)

**Why**: There's no direct Python equivalent to BGLR's full MCMC implementation. The Python version uses regularized regression as approximations:
- Ridge regression approximates BayesA (normal prior with SNP-specific variance)
- Elastic Net approximates BayesB/BayesC (mixture priors)
- Lasso approximates BL (double exponential/Laplace prior)

**Limitation**: These are **simplified approximations**, not true Bayesian MCMC methods. For full Bayesian analysis, you should:
1. Use the R package BGLR
2. Implement proper MCMC in Python (e.g., using `pymc3`, `stan`, or `numpyro`)
3. Use other Bayesian regression packages

---

### 4. **ranger** (R) → **scikit-learn RandomForestRegressor** (Python)

**R Package**: `ranger`
- Used for: Random Forest regression
- Function: `ranger()`

**Python Replacement**: **scikit-learn RandomForestRegressor**
- Package: `scikit-learn`
- Class: `sklearn.ensemble.RandomForestRegressor`
- Functionality: Equivalent random forest implementation

**Why**: `scikit-learn` has a well-established and efficient Random Forest implementation that is functionally equivalent to `ranger` in R.

---

### 5. **xgboost** (R) → **xgboost** (Python) - Same Package!

**R Package**: `xgboost` (R interface)
- Used for: Gradient boosting
- Function: `xgb.train()`, `xgb.DMatrix()`

**Python Replacement**: **xgboost** (Python package)
- Package: `xgboost` (pip install xgboost)
- Functions: `xgb.train()`, `xgb.DMatrix()` - same API!

**Why**: XGBoost is available in both R and Python with very similar APIs. The Python version is the native implementation.

---

### 6. **keras** (R) → **tensorflow/keras** (Python)

**R Package**: `keras` (R interface to TensorFlow)
- Used for: Neural networks (MLP, CNN)
- Functions: `keras_model_sequential()`, `layer_dense()`, etc.

**Python Replacement**: **tensorflow + keras** (Python)
- Package: `tensorflow` (includes Keras)
- Classes: `tensorflow.keras.Sequential`, `tensorflow.keras.layers.Dense`, etc.
- Functionality: Native TensorFlow/Keras implementation

**Why**: TensorFlow/Keras is the native Python implementation. The R `keras` package is just an interface to the Python TensorFlow. Using Python directly gives better performance and more features.

---

### 7. **pls** (R) → **scikit-learn PLSRegression** (Python)

**R Package**: `pls`
- Used for: Partial Least Squares regression
- Function: `plsr()`

**Python Replacement**: **scikit-learn PLSRegression**
- Package: `scikit-learn`
- Class: `sklearn.cross_decomposition.PLSRegression`
- Functionality: Equivalent PLS implementation

**Why**: `scikit-learn` has a comprehensive PLS implementation that matches the functionality of the R `pls` package.

---

### 8. **stats** (R) → **scipy.stats + numpy + pandas** (Python)

**R Package**: `stats` (base R)
- Used for: Statistical functions (`cor()`, `lm()`, `model.matrix()`, etc.)

**Python Replacement**: Multiple packages
- `numpy`: `np.corrcoef()`, `np.var()`, `np.mean()`, etc.
- `scipy.stats`: `stats.linregress()`, statistical distributions
- `pandas`: Data manipulation, `pd.get_dummies()` (equivalent to `model.matrix()`)
- `scipy.linalg`: Linear algebra operations

**Why**: Python's scientific computing stack (NumPy, SciPy, Pandas) provides equivalent functionality to R's `stats` package.

---

### 9. **magrittr** (R) → **Native Python** (No equivalent needed)

**R Package**: `magrittr`
- Used for: Pipe operator (`%>%`)

**Python Replacement**: **Native Python** (no package needed)
- Python has method chaining built-in
- Can use parentheses for chaining, or libraries like `toolz` if needed

**Why**: Python doesn't need a pipe operator - method chaining and function composition are handled differently (and often more naturally) in Python.

---

## Summary Table

| R Package | Python Package | Type | Notes |
|-----------|---------------|------|-------|
| `bigsnpr` | `bed-reader` | Direct replacement | Efficient PLINK file reading |
| `rrBLUP` | `scipy` + custom | Custom implementation | Simplified mixed model solver |
| `BGLR` | `scikit-learn` | Approximation | Uses ridge/lasso/elastic net, not true MCMC |
| `ranger` | `scikit-learn` | Direct replacement | Random Forest |
| `xgboost` | `xgboost` | Same package | Native Python version |
| `keras` | `tensorflow` | Native implementation | TensorFlow/Keras |
| `pls` | `scikit-learn` | Direct replacement | PLS Regression |
| `stats` | `numpy` + `scipy` + `pandas` | Multiple packages | Statistical functions |
| `magrittr` | Native Python | Not needed | Method chaining built-in |

## Key Limitations

1. **BGLR → scikit-learn**: The Python version uses **approximations**, not true Bayesian MCMC. For production Bayesian analysis, consider:
   - Using R BGLR package
   - Implementing MCMC in Python (PyMC3, Stan, NumPyro)
   - Using other Bayesian packages

2. **rrBLUP → scipy**: The Python version uses a **simplified mixed model solver**. For large datasets or complex models, R `rrBLUP` may be more efficient and accurate.

3. **bigsnpr → bed-reader**: Functionally equivalent, but the data structures and some helper functions differ slightly.

## Installation

All Python packages can be installed via pip:

```bash
pip install numpy pandas scipy scikit-learn bed-reader xgboost tensorflow openpyxl
```

Or use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Recommendations

1. **For Bayesian Methods**: If you need true Bayesian MCMC, use the R package BGLR or implement proper MCMC in Python.

2. **For GBLUP**: The Python implementation should work for most cases, but for large datasets (>10,000 individuals) or complex models, consider using R `rrBLUP`.

3. **For Machine Learning**: The Python versions (scikit-learn, xgboost, tensorflow) are excellent and often faster than their R counterparts.

4. **For PLINK File Reading**: `bed-reader` is efficient and works well for most use cases.

