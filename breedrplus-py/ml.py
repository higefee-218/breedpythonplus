"""
Machine learning models for genomic prediction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
import xgboost as xgb
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. MLP and CNN models will not work.")


def _train_ml_model(X_train, y_train, model_type="RF", **kwargs):
    """
    Train a machine learning model for genomic prediction.
    
    Parameters
    ----------
    X_train : ndarray
        Training genotype matrix
    y_train : ndarray
        Training trait values
    model_type : str
        Model type: "RF", "XGB", "MLP", "CNN", or "PLS"
    **kwargs
        Additional arguments for model-specific parameters
        
    Returns
    -------
    dict
        Trained model object and associated parameters
    """
    model_type = model_type.upper()
    
    if model_type == "RF":
        n_trees = kwargs.get('n_trees', 500)
        model_fit = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=42)
        model_fit.fit(X_train, y_train)
        return {'model': model_fit, 'model_type': 'RF'}
        
    elif model_type == "XGB":
        n_rounds = kwargs.get('n_rounds', 100)
        params = {
            'objective': 'reg:squarederror',
            'eta': 0.1,
            'max_depth': 6,
            'eval_metric': 'rmse'
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model_fit = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)
        return {'model': model_fit, 'model_type': 'XGB'}
        
    elif model_type == "MLP":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for MLP models. Install with: pip install tensorflow")
        
        epochs = kwargs.get('epochs', 50)
        
        # Scale X_train
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Clear any existing models
        tf.keras.backend.clear_session()
        
        n_features = X_scaled.shape[1]
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(n_features,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(loss='mse', optimizer=keras.optimizers.RMSprop())
        model.fit(X_scaled, y_train, epochs=epochs, batch_size=32,
                 validation_split=0.1, verbose=0)
        
        return {'model': model, 'scaler': scaler, 'model_type': 'MLP'}
        
    elif model_type == "CNN":
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN models. Install with: pip install tensorflow")
        
        epochs = kwargs.get('epochs', 50)
        
        # Scale X_train
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Clear any existing models
        tf.keras.backend.clear_session()
        
        n_features = X_scaled.shape[1]
        # Reshape for 1D convolution
        X_array = X_scaled.reshape(X_scaled.shape[0], n_features, 1)
        
        model = keras.Sequential([
            layers.Conv1D(filters=32, kernel_size=3, activation='relu', 
                         input_shape=(n_features, 1)),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(loss='mse', optimizer=keras.optimizers.RMSprop())
        model.fit(X_array, y_train, epochs=epochs, batch_size=32,
                 validation_split=0.1, verbose=0)
        
        return {'model': model, 'scaler': scaler, 'model_type': 'CNN', 'n_features': n_features}
        
    elif model_type == "PLS":
        n_comp = kwargs.get('n_comp', min(50, X_train.shape[1]))
        model_fit = PLSRegression(n_components=n_comp)
        model_fit.fit(X_train, y_train)
        return {'model': model_fit, 'model_type': 'PLS', 'n_comp': n_comp}
        
    else:
        raise ValueError(f"model_type must be one of 'RF','XGB','MLP','CNN','PLS'. Got: {model_type}")


def _predict_ml_model(model_fit, X_new, model_type="RF"):
    """
    Predict using a trained ML model.
    
    Parameters
    ----------
    model_fit : dict
        Trained model object
    X_new : ndarray
        New genotype matrix for prediction
    model_type : str
        Model type: "RF", "XGB", "MLP", "CNN", or "PLS"
        
    Returns
    -------
    ndarray
        Vector of predictions
    """
    model_type = model_fit.get('model_type', model_type.upper())
    model = model_fit['model']
    
    if model_type == "RF":
        pred = model.predict(X_new)
        return pred.flatten()
        
    elif model_type == "XGB":
        dpredict = xgb.DMatrix(X_new)
        pred = model.predict(dpredict)
        return pred.flatten()
        
    elif model_type == "PLS":
        n_comp = model_fit.get('n_comp', model.n_components)
        pred = model.predict(X_new)
        return pred.flatten()
        
    elif model_type == "MLP":
        scaler = model_fit['scaler']
        X_scaled = scaler.transform(X_new)
        pred = model.predict(X_scaled, verbose=0)
        return pred.flatten()
        
    elif model_type == "CNN":
        scaler = model_fit['scaler']
        n_features = model_fit['n_features']
        X_scaled = scaler.transform(X_new)
        X_array = X_scaled.reshape(X_scaled.shape[0], n_features, 1)
        pred = model.predict(X_array, verbose=0)
        return pred.flatten()
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_ml_model(pheno, geno, trait_name, id_col_name, model_type="RF", **kwargs):
    """
    Run Machine Learning Model for Genomic Prediction.
    
    Parameters
    ----------
    pheno : pandas.DataFrame
        Data frame with phenotype data
    geno : ndarray or dict
        Numeric matrix of genotype (rownames = individual IDs) or dict with 'genotypes' and 'fam'
    trait_name : str
        Column name of trait in pheno
    id_col_name : str
        Column name of sample IDs in pheno
    model_type : str
        One of "RF", "XGB", "MLP", "CNN", or "PLS"
    **kwargs
        Additional arguments passed to train_ml_model
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'gebv': Data frame with IDs and predicted GEBVs
        - 'model_fit': The trained model object
    """
    # Extract genotype matrix if dict
    if isinstance(geno, dict):
        geno_matrix = geno['genotypes']
        if hasattr(geno['fam'], 'sample.ID'):
            geno_ids = geno['fam']['sample.ID'].values
        else:
            geno_ids = geno['fam']['IID'].values.astype(str)
    else:
        geno_matrix = geno
        if hasattr(geno, 'index'):
            geno_ids = geno.index.values
        else:
            geno_ids = None
    
    # Align genotype rows to phenotype IDs
    if geno_ids is not None:
        pheno = pheno.copy()
        pheno[id_col_name] = pheno[id_col_name].astype(str)
        idx_map = {str(gid): i for i, gid in enumerate(geno_ids)}
        pheno_indices = [idx_map.get(str(pid), None) for pid in pheno[id_col_name]]
        valid_mask = [i is not None for i in pheno_indices]
        pheno = pheno[valid_mask].reset_index(drop=True)
        pheno_indices = [i for i in pheno_indices if i is not None]
        geno_matrix = geno_matrix[pheno_indices, :]
        geno_ids = geno_ids[pheno_indices]
    
    # Ensure alignment
    if geno_ids is not None:
        assert len(pheno) == len(geno_ids), "Phenotype and genotype must have same length"
        assert np.array_equal(pheno[id_col_name].astype(str).values, geno_ids.astype(str)), \
            "Phenotype and genotype IDs must match"
    
    # Extract trait values (remove NAs for training)
    y = pd.to_numeric(pheno[trait_name], errors='coerce').values
    valid_idx = np.where(~np.isnan(y))[0]
    
    if len(valid_idx) == 0:
        raise ValueError("No valid (non-NA) trait values found.")
    
    # Train model on valid observations
    X_train = geno_matrix[valid_idx, :]
    y_train = y[valid_idx]
    
    model_fit = _train_ml_model(X_train, y_train, model_type=model_type, **kwargs)
    
    # Predict for all individuals
    y_pred = _predict_ml_model(model_fit, geno_matrix, model_type=model_type)
    
    gebv_df = pd.DataFrame({
        'ID': pheno[id_col_name].values,
        'gebv': y_pred
    })
    
    return {'gebv': gebv_df, 'model_fit': model_fit}


def run_ml_cv(pheno, geno, trait_name, id_col_name, model_type="RF", k=5, seed=123, **kwargs):
    """
    Run k-fold Cross-Validation for ML genomic prediction.
    
    Parameters
    ----------
    pheno : pandas.DataFrame
        Data frame with phenotype data
    geno : ndarray or dict
        Numeric matrix of genotype (rownames = individual IDs) or dict with 'genotypes' and 'fam'
    trait_name : str
        Column name of trait in pheno
    id_col_name : str
        Column name of sample IDs in pheno
    model_type : str
        One of "RF", "XGB", "MLP", "CNN", or "PLS"
    k : int
        Number of CV folds (default 5)
    seed : int
        Random seed for reproducibility
    **kwargs
        Additional arguments passed to train_ml_model
        
    Returns
    -------
    dict
        Dictionary with:
        - 'cv_gebv': Data frame with predicted GEBVs for all individuals
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
        if hasattr(geno, 'index'):
            geno_ids = geno.index.values
        else:
            geno_ids = None
    
    # Align genotype rows to phenotype IDs
    if geno_ids is not None:
        pheno = pheno.copy()
        pheno[id_col_name] = pheno[id_col_name].astype(str)
        idx_map = {str(gid): i for i, gid in enumerate(geno_ids)}
        pheno_indices = [idx_map.get(str(pid), None) for pid in pheno[id_col_name]]
        valid_mask = [i is not None for i in pheno_indices]
        pheno = pheno[valid_mask].reset_index(drop=True)
        pheno_indices = [i for i in pheno_indices if i is not None]
        geno_matrix = geno_matrix[pheno_indices, :]
        geno_ids = geno_ids[pheno_indices]
    
    # Ensure alignment
    if geno_ids is not None:
        assert len(pheno) == len(geno_ids), "Phenotype and genotype must have same length"
        assert np.array_equal(pheno[id_col_name].astype(str).values, geno_ids.astype(str)), \
            "Phenotype and genotype IDs must match"
    
    # Initialize storage
    y_true = pd.to_numeric(pheno[trait_name], errors='coerce').values
    y_pred = np.full(len(y_true), np.nan)
    
    # Create folds (only on non-NA observations)
    non_na_idx = np.where(~np.isnan(y_true))[0]
    if len(non_na_idx) == 0:
        raise ValueError("No valid (non-NA) trait values found.")
    
    np.random.seed(seed)
    fold_indices = np.random.permutation(len(non_na_idx))
    folds = np.array_split(fold_indices, k)
    
    for i, test_fold_idx in enumerate(folds):
        print(f"CV fold {i+1}/{k}")
        
        test_idx = non_na_idx[test_fold_idx]
        train_idx = np.setdiff1d(non_na_idx, test_idx)
        
        pheno_train = pheno.iloc[train_idx].copy()
        geno_train = geno_matrix[train_idx, :]
        y_train = y_true[train_idx]
        
        # Train on training set
        model_fit = _train_ml_model(geno_train, y_train, model_type=model_type, **kwargs)
        
        # Predict on test set using trained model
        geno_test = geno_matrix[test_idx, :]
        y_pred[test_idx] = _predict_ml_model(model_fit, geno_test, model_type=model_type)
    
    # Compute CV metrics
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    cv_correlation = np.corrcoef(y_true[valid_mask], y_pred[valid_mask])[0, 1]
    cv_mse = np.mean((y_true[valid_mask] - y_pred[valid_mask])**2)
    
    cv_gebv = pd.DataFrame({
        'ID': pheno[id_col_name].values,
        'gebv': y_pred
    })
    
    return {
        'cv_gebv': cv_gebv,
        'cv_correlation': cv_correlation,
        'cv_mse': cv_mse
    }

