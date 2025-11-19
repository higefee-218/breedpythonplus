"""
GBLUP utilities: fit, predict for all, and k-fold CV.
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy import stats
from sklearn.model_selection import KFold, StratifiedKFold
from bed_reader import open_bed


def _compute_G_matrix(bed_file, keep_ind=None, tol_diag=1e-6):
    """
    Compute genomic relationship matrix G.
    
    Parameters
    ----------
    bed_file : str
        Path to .bed file
    keep_ind : array-like, optional
        Indices of individuals to keep. If None, uses all.
    tol_diag : float
        Small value added to diagonal for numerical stability.
        
    Returns
    -------
    G : ndarray
        Genomic relationship matrix
    """
    bed = open_bed(bed_file)
    n_ind = bed.shape[0]
    
    if keep_ind is None:
        keep_ind = np.arange(n_ind)
    else:
        keep_ind = np.asarray(keep_ind)
    
    # Read genotypes for selected individuals
    genotypes = bed.read(index=np.s_[keep_ind, :])
    
    # Compute allele frequencies
    p = np.nanmean(genotypes, axis=0) / 2.0
    p = np.clip(p, 0.01, 0.99)  # Avoid division by zero
    
    # Standardize genotypes: (X - 2p) / sqrt(2p(1-p))
    # This is the VanRaden G matrix computation
    Z = genotypes - 2 * p
    Z = Z / np.sqrt(2 * p * (1 - p))
    Z = np.nan_to_num(Z, nan=0.0)  # Replace NaN with 0
    
    # Compute G = Z Z' / m (where m is number of SNPs)
    m = genotypes.shape[1]
    G = np.dot(Z, Z.T) / m
    
    # Add small value to diagonal for numerical stability
    np.fill_diagonal(G, np.diag(G) + tol_diag)
    
    return G


def _fit_mixed_model(y, X, K):
    """
    Fit mixed model using simplified approach (approximation of rrBLUP::mixed.solve).
    
    Uses iterative method to estimate variance components and solve mixed model equations.
    This is a simplified implementation - for production use with large datasets,
    consider using specialized mixed model libraries.
    
    Parameters
    ----------
    y : ndarray
        Phenotype vector
    X : ndarray
        Design matrix for fixed effects
    K : ndarray
        Kinship/genomic relationship matrix
        
    Returns
    -------
    dict
        Dictionary with 'Vu' (genetic variance), 'Ve' (residual variance),
        'beta' (fixed effects), 'u' (random effects/BLUP)
    """
    n = len(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    p = X.shape[1]
    
    # Initial estimates: use simple method of moments
    # Start with lambda = Ve/Vu ratio
    lambda_val = 1.0
    
    # Iterate to find optimal lambda
    max_iter = 20
    tolerance = 1e-4
    
    for iteration in range(max_iter):
        try:
            # Build V = K*Vu + I*Ve = Vu*(K + lambda*I) where lambda = Ve/Vu
            V = K + lambda_val * np.eye(n)
            
            # Add small regularization for numerical stability
            V += 1e-6 * np.eye(n)
            
            # Solve for beta using generalized least squares
            # beta = (X' * V^-1 * X)^-1 * X' * V^-1 * y
            try:
                V_inv = solve(V, np.eye(n))
                XtVinvX = np.dot(X.T, np.dot(V_inv, X))
                XtVinvy = np.dot(X.T, np.dot(V_inv, y))
                beta_hat = solve(XtVinvX, XtVinvy)
            except:
                # Fallback: use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(V)
                    Linv = solve(L, np.eye(n))
                    V_inv = np.dot(Linv.T, Linv)
                    XtVinvX = np.dot(X.T, np.dot(V_inv, X))
                    XtVinvy = np.dot(X.T, np.dot(V_inv, y))
                    beta_hat = solve(XtVinvX, XtVinvy)
                except:
                    # Last resort: OLS
                    XtX = np.dot(X.T, X)
                    Xty = np.dot(X.T, y)
                    beta_hat = solve(XtX, Xty)
            
            # Compute fitted values and residuals
            y_fitted = np.dot(X, beta_hat)
            residuals = y - y_fitted
            
            # Solve for u: u = K * inv(K + lambda*I) * residuals
            # This gives BLUP of random effects
            try:
                A = K + lambda_val * np.eye(n) + 1e-6 * np.eye(n)
                u = solve(A, np.dot(K, residuals))
            except:
                # Use Cholesky
                try:
                    L = np.linalg.cholesky(A)
                    b = np.dot(K, residuals)
                    u = solve(L, b)
                    u = solve(L.T, u)
                except:
                    # Simple approximation
                    u = np.dot(K, residuals) / (1 + lambda_val)
            
            # Estimate variance components using method of moments
            # Ve = residual variance
            pred_error = residuals - u
            Ve_new = np.var(pred_error)
            Ve_new = max(Ve_new, 1e-6)
            
            # Vu = genetic variance (simplified estimate)
            # Use u' * u / trace(K) as approximation
            try:
                Vu_new = np.dot(u, u) / np.trace(K)
            except:
                Vu_new = np.var(u)
            Vu_new = max(Vu_new, 1e-6)
            
            # Update lambda
            lambda_new = Ve_new / Vu_new if Vu_new > 0 else 1.0
            
            # Check convergence
            if abs(lambda_new - lambda_val) / (lambda_val + 1e-6) < tolerance:
                lambda_val = lambda_new
                break
            
            lambda_val = 0.7 * lambda_val + 0.3 * lambda_new  # Damped update
            
        except Exception as e:
            # If iteration fails, break and use current values
            break
    
    # Final fit with converged (or last) lambda
    try:
        V = K + lambda_val * np.eye(n) + 1e-6 * np.eye(n)
        try:
            V_inv = solve(V, np.eye(n))
            XtVinvX = np.dot(X.T, np.dot(V_inv, X))
            XtVinvy = np.dot(X.T, np.dot(V_inv, y))
            beta_hat = solve(XtVinvX, XtVinvy)
        except:
            L = np.linalg.cholesky(V)
            Linv = solve(L, np.eye(n))
            V_inv = np.dot(Linv.T, Linv)
            XtVinvX = np.dot(X.T, np.dot(V_inv, X))
            XtVinvy = np.dot(X.T, np.dot(V_inv, y))
            beta_hat = solve(XtVinvX, XtVinvy)
        
        residuals = y - np.dot(X, beta_hat)
        A = K + lambda_val * np.eye(n) + 1e-6 * np.eye(n)
        u = solve(A, np.dot(K, residuals))
        
        # Final variance estimates
        pred_error = residuals - u
        Ve = max(np.var(pred_error), 1e-6)
        Vu = max(np.dot(u, u) / max(np.trace(K), 1.0), 1e-6)
        
    except Exception as e:
        # Fallback: very simple approach
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y)
        beta_hat = solve(XtX, Xty)
        residuals = y - np.dot(X, beta_hat)
        A = K + lambda_val * np.eye(n) + 1e-6 * np.eye(n)
        u = solve(A, np.dot(K, residuals))
        Vu = max(np.var(u), 1e-6)
        Ve = max(lambda_val * Vu, 1e-6)
    
    return {
        'Vu': float(Vu),
        'Ve': float(Ve),
        'beta': beta_hat.flatten() if beta_hat.ndim > 1 and beta_hat.shape[1] == 1 else beta_hat,
        'u': u.flatten()
    }


def run_gblup(qc_results, trait_name, id_col_name, fixed_effects=None,
              drop_missing=True, predict_all=False, tol_diag=1e-6):
    """
    Run GBLUP and optionally predict for all individuals.
    
    Parameters
    ----------
    qc_results : dict
        Dictionary with 'snp_obj' and 'pheno' keys
    trait_name : str
        Trait column name
    id_col_name : str
        ID column name
    fixed_effects : list of str, optional
        Column names for fixed effects. If None, uses intercept only.
        Note: R version uses formula syntax (e.g., ~1, ~sex+age), but Python
        uses a list of column names (e.g., ["sex", "age"]). For intercept only,
        pass None or empty list.
    drop_missing : bool
        If True, drops individuals with NA phenotype for fitting.
    predict_all : bool
        If True, compute GEBVs for all individuals.
    tol_diag : float
        Small ridge added to diagonal for numerical stability.
        
    Returns
    -------
    dict
        Dictionary with 'gebv_obs', 'variances', 'model_fit', 'G', 
        'keep_ind', 'ids_full', and optionally 'gebv_all'
    """
    if qc_results is None:
        raise ValueError("qc_results must be provided.")
    if 'snp_obj' not in qc_results:
        raise ValueError("qc_results$snp_obj not found.")
    if 'pheno' not in qc_results:
        raise ValueError("qc_results$pheno not found.")
    
    snp_obj = qc_results['snp_obj']
    pheno = qc_results['pheno']
    
    if trait_name not in pheno.columns:
        raise ValueError(f"trait_name '{trait_name}' not found in pheno.")
    if id_col_name not in pheno.columns:
        raise ValueError(f"id_col_name '{id_col_name}' not found in pheno.")
    
    # Get keep_ind (default to all individuals)
    n_total = len(snp_obj['fam'])
    keep_ind = qc_results.get('keep_ind', np.arange(n_total))
    keep_ind = np.asarray(keep_ind)
    
    # Subset phenotype to keep_ind rows
    ph_sub = pheno.iloc[keep_ind].copy()
    ph_sub[trait_name] = pd.to_numeric(ph_sub[trait_name], errors='coerce')
    y_full = ph_sub[trait_name].values
    ids_full = ph_sub[id_col_name].values
    
    # Build X design matrix
    if fixed_effects is None:
        X_full = np.ones((len(ph_sub), 1))
    else:
        # Create design matrix from fixed effects columns
        X_cols = [np.ones(len(ph_sub))]
        for fe in fixed_effects:
            if fe in ph_sub.columns:
                X_cols.append(pd.get_dummies(ph_sub[fe], drop_first=True).values)
        X_full = np.hstack(X_cols)
    
    # Count missing
    na_count = np.sum(np.isnan(y_full))
    
    # Compute G matrix
    bedfile_path = snp_obj['bedfile']
    if bedfile_path is None:
        raise ValueError("snp_obj$bedfile is missing. Re-run load_plink() that sets snp_obj$bedfile.")
    
    print("Computing G matrix (this may take some seconds)...")
    G_full = _compute_G_matrix(bedfile_path, keep_ind=keep_ind, tol_diag=tol_diag)
    print(f"G computed. Dimensions: {G_full.shape}")
    
    # Decide training set
    if na_count > 0:
        if drop_missing:
            obs_idx = np.where(~np.isnan(y_full))[0]
            print(f"Dropping {na_count} missing phenotypes; fitting on {len(obs_idx)} observed individuals.")
        else:
            raise ValueError("Missing phenotypes present. Set drop_missing = True to fit only on observed individuals.")
    else:
        obs_idx = np.arange(len(y_full))
    
    # Observed data subsets
    y_obs = y_full[obs_idx]
    X_obs = X_full[obs_idx, :]
    ids_obs = ids_full[obs_idx]
    
    # Subset G to observed individuals for fitting
    G_obs = G_full[np.ix_(obs_idx, obs_idx)]
    
    # Fit mixed model
    print("Fitting mixed model...")
    fit = _fit_mixed_model(y_obs, X_obs, G_obs)
    print("Model fitted.")
    
    # Extract variance components and heritability
    sigma_g = fit['Vu']
    sigma_e = fit['Ve']
    h2 = sigma_g / (sigma_g + sigma_e) if (sigma_g + sigma_e) > 0 else 0.0
    variances = {'sigma_g': sigma_g, 'sigma_e': sigma_e, 'h2': h2}
    
    # GEBVs for observed individuals
    u_obs = fit['u']
    gebv_obs_df = pd.DataFrame({
        id_col_name: ids_obs,
        'gebv': u_obs
    })
    
    out = {
        'gebv_obs': gebv_obs_df,
        'variances': variances,
        'model_fit': fit,
        'G': G_full,
        'keep_ind': keep_ind,
        'ids_full': ids_full
    }
    
    # If predict_all requested, compute GEBVs for all individuals
    if predict_all:
        print("Computing GEBVs for all individuals (observed + unobserved)...")
        
        ratio = sigma_e / sigma_g if sigma_g > 0 else 1.0
        A = G_obs + ratio * np.eye(len(obs_idx))
        beta_hat = fit['beta']
        if X_obs.ndim == 1:
            X_obs = X_obs.reshape(-1, 1)
        rvec = y_obs - np.dot(X_obs, beta_hat.reshape(-1, 1)).flatten()
        
        # Precompute invA %*% rvec
        invA_r = solve(A, rvec)
        
        # G_all_obs: rows = all individuals, cols = observed individuals
        G_all_obs = G_full[:, obs_idx]
        
        u_all = np.dot(G_all_obs, invA_r)
        
        # Build dataframe for all individuals
        gebv_all_df = pd.DataFrame({
            id_col_name: ids_full,
            'gebv': u_all
        })
        
        out['gebv_all'] = gebv_all_df
        print("GEBVs for all individuals computed.")
    
    return out


def cv_gblup(qc_results, trait_name, id_col_name, k=5, fixed_effects=None,
             seed=2025, stratify_by=None, drop_missing=True, return_fold_preds=True):
    """
    Cross-validate GBLUP (k-fold).
    
    Parameters
    ----------
    qc_results : dict
        Same as run_gblup
    trait_name : str
        Trait column name
    id_col_name : str
        ID column name
    k : int
        Number of folds (default 5)
    fixed_effects : list of str, optional
        Column names for fixed effects. If None, uses intercept only.
        Note: R version uses formula syntax (e.g., ~1, ~sex+age), but Python
        uses a list of column names (e.g., ["sex", "age"]). For intercept only,
        pass None or empty list.
    seed : int
        Random seed for fold assignment
    stratify_by : str, optional
        Column name to stratify folds by
    drop_missing : bool
        If True, individuals with NA trait are removed before fold assignment
    return_fold_preds : bool
        If True, include predictions for each test set row in output
        
    Returns
    -------
    dict
        Dictionary containing 'fold_results', 'overall', and optionally 'predictions'
    """
    if qc_results is None:
        raise ValueError("qc_results must be provided.")
    if 'snp_obj' not in qc_results:
        raise ValueError("qc_results$snp_obj not found.")
    if 'pheno' not in qc_results:
        raise ValueError("qc_results$pheno not found.")
    
    snp_obj = qc_results['snp_obj']
    pheno = qc_results['pheno']
    
    if trait_name not in pheno.columns:
        raise ValueError(f"trait_name '{trait_name}' not found in pheno.")
    if id_col_name not in pheno.columns:
        raise ValueError(f"id_col_name '{id_col_name}' not found in pheno.")
    
    # keep_ind default
    n_total = len(snp_obj['fam'])
    keep_ind = qc_results.get('keep_ind', np.arange(n_total))
    keep_ind = np.asarray(keep_ind)
    
    ph_sub = pheno.iloc[keep_ind].copy()
    ph_sub[trait_name] = pd.to_numeric(ph_sub[trait_name], errors='coerce')
    ids_all = ph_sub[id_col_name].values
    y_all = ph_sub[trait_name].values
    
    # Remove rows with NA if requested
    if drop_missing:
        valid_idx = np.where(~np.isnan(y_all))[0]
        if len(valid_idx) < len(y_all):
            print(f"Removing {len(y_all) - len(valid_idx)} rows with missing trait before CV.")
    else:
        valid_idx = np.arange(len(y_all))
        if np.any(np.isnan(y_all)):
            raise ValueError("Missing phenotypes present and drop_missing = FALSE.")
    
    # Prepare folds
    np.random.seed(seed)
    n_valid = len(valid_idx)
    
    if stratify_by is not None and stratify_by in ph_sub.columns:
        # Stratified k-fold
        strata = ph_sub[stratify_by].values[valid_idx]
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        fold_splits = list(kf.split(valid_idx, strata))
    else:
        # Regular k-fold
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        fold_splits = list(kf.split(valid_idx))
    
    # Compute full G once
    bedfile_path = snp_obj['bedfile']
    if bedfile_path is None:
        raise ValueError("snp_obj$bedfile missing; re-run load_plink() that sets it.")
    
    print("Computing G matrix for CV...")
    G_full = _compute_G_matrix(bedfile_path, keep_ind=keep_ind, tol_diag=1e-6)
    
    # Precompute X design matrix
    if fixed_effects is None:
        X_full = np.ones((len(ph_sub), 1))
    else:
        X_cols = [np.ones(len(ph_sub))]
        for fe in fixed_effects:
            if fe in ph_sub.columns:
                X_cols.append(pd.get_dummies(ph_sub[fe], drop_first=True).values)
        X_full = np.hstack(X_cols)
    
    # Storage for fold metrics and predictions
    fold_results = []
    all_preds = []
    
    for fold, (train_pos, test_pos) in enumerate(fold_splits, 1):
        if len(test_pos) == 0:
            print(f"Warning: Fold {fold} has zero test samples; skipping.")
            continue
        
        # Map to global indices
        train_idx_global = valid_idx[train_pos]
        test_idx_global = valid_idx[test_pos]
        
        # Build training subsets
        y_train = y_all[train_idx_global]
        X_train = X_full[train_idx_global, :]
        ids_train = ids_all[train_idx_global]
        
        # G matrices
        G_train = G_full[np.ix_(train_idx_global, train_idx_global)]
        G_test_train = G_full[np.ix_(test_idx_global, train_idx_global)]
        
        # Fit model on training set
        fit_tr = _fit_mixed_model(y_train, X_train, G_train)
        
        # Extract variances and beta
        Vu = fit_tr['Vu']
        Ve = fit_tr['Ve']
        beta_hat = fit_tr['beta']
        r_train = y_train - np.dot(X_train, beta_hat.reshape(-1, 1)).flatten()
        
        # Solve A = (G_train + (Ve/Vu) I)
        ratio = Ve / Vu if Vu > 0 else 1.0
        A = G_train + ratio * np.eye(len(train_idx_global))
        invA_r = solve(A, r_train)
        
        # Predict u for test samples
        u_test = np.dot(G_test_train, invA_r)
        
        # Predicted phenotype = X_test %*% beta_hat + u_test
        X_test = X_full[test_idx_global, :]
        yhat_test = np.dot(X_test, beta_hat.reshape(-1, 1)).flatten() + u_test
        yobs_test = y_all[test_idx_global]
        ids_test = ids_all[test_idx_global]
        
        # Metrics
        if len(np.unique(yobs_test)) > 1:
            cor_val = np.corrcoef(yhat_test, yobs_test)[0, 1]
        else:
            cor_val = np.nan
        rmse_val = np.sqrt(np.mean((yhat_test - yobs_test)**2))
        
        # Bias: regression slope of observed ~ predicted
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(yhat_test, yobs_test)
            bias_slope = slope
        except:
            bias_slope = np.nan
        
        fold_results.append({
            'fold': fold,
            'n_train': len(y_train),
            'n_test': len(yobs_test),
            'cor': cor_val,
            'rmse': rmse_val,
            'bias_slope': bias_slope,
            'Vu': Vu,
            'Ve': Ve,
            'h2_est': Vu / (Vu + Ve) if (Vu + Ve) > 0 else 0.0
        })
        
        if return_fold_preds:
            all_preds.append(pd.DataFrame({
                'ID': ids_test,
                'yobs': yobs_test,
                'yhat': yhat_test,
                'fold': fold
            }))
    
    # Combine fold results
    fold_df = pd.DataFrame(fold_results)
    overall = {
        'mean_cor': fold_df['cor'].mean(),
        'mean_rmse': fold_df['rmse'].mean(),
        'mean_bias_slope': fold_df['bias_slope'].mean(),
        'mean_h2': fold_df['h2_est'].mean()
    }
    
    out = {
        'fold_results': fold_df,
        'overall': pd.Series(overall)
    }
    if return_fold_preds:
        out['predictions'] = pd.concat(all_preds, ignore_index=True)
    
    return out

