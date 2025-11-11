"""
Example Workflow for breedrplus Package
This script demonstrates how to use the package with the example dataset
"""

import os
import pandas as pd
import numpy as np
from breedrplus import (
    load_plink, run_gblup, cv_gblup,
    run_bglr, run_bglr_cv,
    run_ml_model, run_ml_cv
)

# ============================================================================
# Step 1: Load the example data
# ============================================================================

# Get paths to example data files
# Adjust these paths to match your data location
base_dir = "inst/extdata"
bed_file = os.path.join(base_dir, "500ind.30K.bed")
bim_file = os.path.join(base_dir, "500ind.30K.bim")
fam_file = os.path.join(base_dir, "500ind.30K.fam")
pheno_file = os.path.join(base_dir, "500_ind_pheno.xlsx")

# Check if files exist
if not os.path.exists(bed_file):
    print(f"Warning: {bed_file} not found. Please adjust the path.")
    print("Example workflow will show the structure but may not run.")

# Load phenotype data
try:
    pheno = pd.read_excel(pheno_file)
    print("Phenotype data loaded successfully.")
    print(f"Phenotype shape: {pheno.shape}")
    print(pheno.head())
    print(pheno.dtypes)
except Exception as e:
    print(f"Error loading phenotype file: {e}")
    print("Creating dummy phenotype data for demonstration...")
    # Create dummy data for demonstration
    pheno = pd.DataFrame({
        'ID': [f'IND{i+1}' for i in range(500)],
        'Trait': np.random.normal(0, 1, 500)
    })

# ============================================================================
# Step 2: Load PLINK files and align phenotypes
# ============================================================================

# Load PLINK files into NumPy format
# Note: You'll need to identify the ID column name in your phenotype file
# Replace "ID" with the actual column name containing individual IDs

try:
    data = load_plink(
        bed_file=bed_file,
        bim_file=bim_file,
        fam_file=fam_file,
        pheno=pheno,
        id_col_name="ID",  # Update this to match your phenotype file
        impute_method="mode"
    )
    
    print("\nData loaded successfully!")
    print(f"Genotype shape: {data['snp_obj']['genotypes'].shape}")
    print(f"Phenotype shape: {data['pheno'].shape if data['pheno'] is not None else 'None'}")
    if data['pheno'] is not None:
        print(data['pheno'].head())
except Exception as e:
    print(f"Error loading PLINK files: {e}")
    print("Skipping to next sections with dummy data...")
    # Create dummy data structure
    data = {
        'snp_obj': {
            'genotypes': np.random.randint(0, 3, (500, 30000)),
            'fam': pd.DataFrame({
                'sample.ID': [f'IND{i+1}' for i in range(500)]
            }),
            'bedfile': bed_file
        },
        'pheno': pheno
    }

# ============================================================================
# Step 3: Run GBLUP
# ============================================================================

# Identify the trait column name (update this to match your data)
trait_name = "Trait"  # Update this to your actual trait column name

try:
    # Fit GBLUP model
    print("\n" + "="*60)
    print("Running GBLUP...")
    print("="*60)
    gblup_result = run_gblup(
        qc_results=data,
        trait_name=trait_name,
        id_col_name="ID",  # Update if different
        fixed_effects=None,  # Use intercept only, or provide list of column names
        drop_missing=True,
        predict_all=True
    )
    
    # View results
    print("\nVariance components:")
    print(gblup_result['variances'])
    print("\nGEBVs for observed individuals:")
    print(gblup_result['gebv_obs'].head())
    if 'gebv_all' in gblup_result:
        print("\nGEBVs for all individuals:")
        print(gblup_result['gebv_all'].head())
except Exception as e:
    print(f"Error running GBLUP: {e}")
    import traceback
    traceback.print_exc()

# Cross-validation
try:
    print("\n" + "="*60)
    print("Running GBLUP Cross-Validation...")
    print("="*60)
    cv_gblup_result = cv_gblup(
        qc_results=data,
        trait_name=trait_name,
        id_col_name="ID",
        k=5,
        seed=2025
    )
    
    print("\nOverall CV results:")
    print(cv_gblup_result['overall'])
    print("\nFold results:")
    print(cv_gblup_result['fold_results'])
except Exception as e:
    print(f"Error running GBLUP CV: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Step 4: Run Bayesian Methods
# ============================================================================

# Extract genotype matrix for Bayesian methods
try:
    print("\n" + "="*60)
    print("Running Bayesian Methods...")
    print("="*60)
    geno_matrix = data['snp_obj']['genotypes']
    
    # Fit Bayesian model
    bayes_result = run_bglr(
        obj=data['snp_obj'],
        pheno=data['pheno'],
        trait_name=trait_name,
        id_col_name="ID",
        model_type="BayesA",
        n_iter=5000,
        burn_in=1000
    )
    
    print("\nBayesian results:")
    print(bayes_result['gebv'].head())
    print(f"\nResidual variance: {bayes_result['residual_var']}")
    
    # Cross-validation
    print("\nRunning Bayesian CV...")
    bayes_cv = run_bglr_cv(
        geno=geno_matrix,
        pheno=data['pheno'],
        trait_name=trait_name,
        id_col_name="ID",
        model_type="BayesA",
        n_iter=5000,
        burn_in=1000,
        k_folds=5,
        seed=123
    )
    
    print(f"\nCV Correlation: {bayes_cv['cv_correlation']}")
    print(f"CV MSE: {bayes_cv['cv_mse']}")
except Exception as e:
    print(f"Error running Bayesian methods: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Step 5: Run Machine Learning Models
# ============================================================================

try:
    print("\n" + "="*60)
    print("Running Machine Learning Models...")
    print("="*60)
    
    # Random Forest
    print("\nTraining Random Forest...")
    ml_rf = run_ml_model(
        pheno=data['pheno'],
        geno=geno_matrix,
        trait_name=trait_name,
        id_col_name="ID",
        model_type="RF",
        n_trees=500
    )
    
    print("\nRandom Forest results:")
    print(ml_rf['gebv'].head())
    
    # Cross-validation
    print("\nRunning Random Forest CV...")
    ml_cv_rf = run_ml_cv(
        pheno=data['pheno'],
        geno=geno_matrix,
        trait_name=trait_name,
        id_col_name="ID",
        model_type="RF",
        k=5,
        seed=123
    )
    
    print(f"\nCV Correlation: {ml_cv_rf['cv_correlation']}")
    print(f"CV MSE: {ml_cv_rf['cv_mse']}")
    
    # XGBoost
    print("\nRunning XGBoost CV...")
    ml_cv_xgb = run_ml_cv(
        pheno=data['pheno'],
        geno=geno_matrix,
        trait_name=trait_name,
        id_col_name="ID",
        model_type="XGB",
        k=5,
        seed=123,
        n_rounds=100
    )
    
    print(f"\nXGBoost CV Correlation: {ml_cv_xgb['cv_correlation']}")
    print(f"XGBoost CV MSE: {ml_cv_xgb['cv_mse']}")
except Exception as e:
    print(f"Error running ML models: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Notes:
# ============================================================================
print("\n" + "="*60)
print("Notes:")
print("="*60)
print("1. Update 'ID' and 'Trait' column names to match your actual data")
print("2. Adjust model parameters (n_iter, burn_in, k, etc.) as needed")
print("3. For large datasets, consider using predict_all = False in run_gblup()")
print("4. Machine learning models may require more computational resources")
print("5. TensorFlow is required for MLP and CNN models")

