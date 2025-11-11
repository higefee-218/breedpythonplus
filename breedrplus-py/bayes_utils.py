"""
Bayesian genomic prediction utilities using Bayesian regression.
Note: This is a simplified implementation. BGLR in R has more sophisticated
MCMC sampling. This implementation uses a simplified Bayesian approach.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler


def _bayesian_regression(X, y, model_type="BayesA", n_iter=5000, burn_in=1000):
    """
    Simplified Bayesian regression (approximation of BGLR).
    
    For a full BGLR equivalent, you would need MCMC sampling.
    This is a simplified version using variational inference or 
    ridge regression approximations.
    
    Parameters
    ----------
    X : ndarray
        Genotype matrix (scaled)
    y : ndarray
        Phenotype vector
    model_type : str
        Model type: "BayesA", "BayesB", "BayesC", "BL" (Bayesian Lasso)
    n_iter : int
        Number of iterations (not used in simplified version)
    burn_in : int
        Burn-in iterations (not used in simplified version)
        
    Returns
    -------
    dict
        Dictionary with 'b' (SNP effects), 'yHat' (predicted values), 'varE' (residual variance)
    """
    n, p = X.shape
    
    # Scale X if not already scaled
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Simplified Bayesian regression using ridge regression with different penalties
    # This is an approximation - full BGLR uses MCMC
    
    if model_type == "BayesA":
        # BayesA: each SNP has its own variance (inverse gamma prior)
        # Approximate with adaptive ridge regression
        alpha = 1.0  # regularization parameter
        # Use ridge regression as approximation
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X_scaled, y)
        b = model.coef_
        yHat = model.predict(X_scaled)
        
    elif model_type == "BayesB":
        # BayesB: mixture of point mass at zero and normal
        # Approximate with elastic net
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=False, max_iter=10000)
        model.fit(X_scaled, y)
        b = model.coef_
        yHat = model.predict(X_scaled)
        
    elif model_type == "BayesC":
        # BayesC: mixture similar to BayesB but with common variance
        # Approximate with elastic net
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(alpha=0.5, l1_ratio=0.7, fit_intercept=False, max_iter=10000)
        model.fit(X_scaled, y)
        b = model.coef_
        yHat = model.predict(X_scaled)
        
    elif model_type == "BL":
        # Bayesian Lasso: double exponential prior
        # Use Lasso regression
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.1, fit_intercept=False, max_iter=10000)
        model.fit(X_scaled, y)
        b = model.coef_
        yHat = model.predict(X_scaled)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be one of 'BayesA', 'BayesB', 'BayesC', 'BL'")
    
    # Estimate residual variance
    residuals = y - yHat
    varE = np.var(residuals)
    
    return {
        'b': b,
        'yHat': yHat,
        'varE': varE,
        'scaler': scaler
    }


def run_bglr(obj, pheno, trait_name, id_col_name, model_type="BayesA",
             n_iter=5000, burn_in=1000):
    """
    Run Bayesian Genomic Prediction using simplified Bayesian regression.
    
    Parameters
    ----------
    obj : dict
        Dictionary with 'genotypes' (2D array) and 'fam' (DataFrame)
    pheno : pandas.DataFrame
        Data frame containing phenotype data with individual IDs
    trait_name : str
        Character string naming the trait column in pheno
    id_col_name : str
        Character string naming the ID column in pheno
    model_type : str
        Bayesian model type. Options: "BayesA", "BayesB", "BayesC", "BL" (Bayesian Lasso). Default "BayesA"
    n_iter : int
        Number of MCMC iterations (not used in simplified version). Default 5000
    burn_in : int
        Number of burn-in iterations (not used in simplified version). Default 1000
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'gebv': Data frame with individual IDs and genomic estimated breeding values
        - 'snp_effects': Data frame with SNP effects
        - 'residual_var': Residual variance estimate
        - 'model_fit': The full model fit object
    """
    # Extract numeric genotype matrix
    if isinstance(obj, dict):
        geno = obj['genotypes']
        if hasattr(obj['fam'], 'sample.ID'):
            geno_ids = obj['fam']['sample.ID'].values
        else:
            geno_ids = obj['fam']['IID'].values.astype(str)
    else:
        # Assume it's a numpy array
        geno = obj
        geno_ids = None
    
    # Create SNP names if not available
    n_snps = geno.shape[1]
    snp_names = [f"SNP{i+1}" for i in range(n_snps)]
    
    # Align phenotype to genotype IDs
    if geno_ids is not None:
        idx = []
        for gid in geno_ids:
            match_idx = pheno[pheno[id_col_name].astype(str) == str(gid)].index
            if len(match_idx) > 0:
                idx.append(match_idx[0])
            else:
                idx.append(None)
        
        # Filter out None indices
        valid_idx = [i for i, x in enumerate(idx) if x is not None]
        geno = geno[valid_idx, :]
        geno_ids = geno_ids[valid_idx]
        idx = [idx[i] for i in valid_idx]
        
        pheno_aligned = pheno.iloc[idx].reset_index(drop=True)
        
        if not np.array_equal(pheno_aligned[id_col_name].astype(str).values, geno_ids.astype(str)):
            print("Warning: Genotype IDs and phenotype IDs may not match perfectly.")
    else:
        # Assume they're already aligned
        pheno_aligned = pheno.copy()
        geno_ids = pheno[id_col_name].values
    
    # Trait vector
    y = pd.to_numeric(pheno_aligned[trait_name], errors='coerce').values
    
    # Remove missing values
    valid_mask = ~np.isnan(y)
    y = y[valid_mask]
    geno = geno[valid_mask, :]
    geno_ids = geno_ids[valid_mask]
    
    # Fit Bayesian model
    fit = _bayesian_regression(geno, y, model_type=model_type, n_iter=n_iter, burn_in=burn_in)
    
    # Extract SNP effects
    snp_effects = fit['b']
    if len(snp_effects) != n_snps:
        # If some SNPs were dropped, pad with zeros
        snp_effects_full = np.zeros(n_snps)
        snp_effects_full[:len(snp_effects)] = snp_effects
        snp_effects = snp_effects_full
    
    snp_effects_df = pd.DataFrame({
        'SNP': snp_names[:len(snp_effects)],
        'Effect': snp_effects
    })
    
    # Extract GEBVs
    gebv = fit['yHat']
    gebv_df = pd.DataFrame({
        'ID': geno_ids,
        'GEBV': gebv
    })
    
    return {
        'gebv': gebv_df,
        'snp_effects': snp_effects_df,
        'residual_var': fit['varE'],
        'model_fit': fit
    }


def run_bglr_cv(geno, pheno, trait_name, id_col_name, model_type="BayesA",
                n_iter=5000, burn_in=1000, k_folds=5, seed=123):
    """
    Cross-Validate Bayesian Genomic Prediction.
    
    Parameters
    ----------
    geno : ndarray or dict
        Numeric genotype matrix with rownames as individual IDs, or dict with 'genotypes' and 'fam'
    pheno : pandas.DataFrame
        Data frame containing phenotype data with individual IDs
    trait_name : str
        Character string naming the trait column in pheno
    id_col_name : str
        Character string naming the ID column in pheno
    model_type : str
        Bayesian model type. Options: "BayesA", "BayesB", "BayesC", "BL". Default "BayesA"
    n_iter : int
        Number of MCMC iterations. Default 5000
    burn_in : int
        Number of burn-in iterations. Default 1000
    k_folds : int
        Number of cross-validation folds. Default 5
    seed : int
        Random seed for reproducibility. Default 123
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'y_true': Observed trait values
        - 'y_pred': Predicted trait values from cross-validation
        - 'cv_correlation': Pearson correlation between observed and predicted values
        - 'cv_mse': Mean squared error
    """
    np.random.seed(seed)
    
    # Extract genotype matrix if dict
    if isinstance(geno, dict):
        geno_matrix = geno['genotypes']
        if hasattr(geno['fam'], 'sample.ID'):
            geno_ids = geno['fam']['sample.ID'].values
        else:
            geno_ids = geno['fam']['IID'].values.astype(str)
    else:
        geno_matrix = geno
        # Try to get IDs from rownames if available
        if hasattr(geno, 'index'):
            geno_ids = geno.index.values
        else:
            geno_ids = None
    
    # Align phenotype and genotype
    if geno_ids is not None:
        # Match by ID
        pheno = pheno.copy()
        pheno[id_col_name] = pheno[id_col_name].astype(str)
        idx_map = {str(gid): i for i, gid in enumerate(geno_ids)}
        pheno_indices = [idx_map.get(str(pid), None) for pid in pheno[id_col_name]]
        valid_mask = [i is not None for i in pheno_indices]
        pheno = pheno[valid_mask].reset_index(drop=True)
        pheno_indices = [i for i in pheno_indices if i is not None]
        geno_matrix = geno_matrix[pheno_indices, :]
        geno_ids = geno_ids[pheno_indices]
    
    y_true = pd.to_numeric(pheno[trait_name], errors='coerce').values
    n = len(y_true)
    
    # Remove missing values
    valid_mask = ~np.isnan(y_true)
    y_true = y_true[valid_mask]
    geno_matrix = geno_matrix[valid_mask, :]
    geno_ids = geno_ids[valid_mask] if geno_ids is not None else None
    
    n = len(y_true)
    
    # Create folds
    np.random.seed(seed)
    fold_indices = np.random.permutation(n)
    folds = np.array_split(fold_indices, k_folds)
    
    # Store predictions
    y_pred = np.full(n, np.nan)
    
    for f, test_idx in enumerate(folds):
        print(f"Running fold {f+1} of {k_folds}")
        
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        
        # Training data
        y_train = y_true[train_idx]
        geno_train = geno_matrix[train_idx, :]
        
        # Fit BGLR
        fit = _bayesian_regression(geno_train, y_train, model_type=model_type, 
                                   n_iter=n_iter, burn_in=burn_in)
        
        # Test data: scale using training scaler
        scaler = fit['scaler']
        geno_test = scaler.transform(geno_matrix[test_idx, :])
        
        # Predict
        b = fit['b']
        y_pred[test_idx] = np.dot(geno_test, b)
    
    # Compute CV metrics
    cv_cor = np.corrcoef(y_true, y_pred)[0, 1]
    cv_mse = np.mean((y_true - y_pred)**2)
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'cv_correlation': cv_cor,
        'cv_mse': cv_mse
    }

