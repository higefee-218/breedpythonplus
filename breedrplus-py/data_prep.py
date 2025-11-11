"""
Data preparation utilities for loading PLINK files and aligning phenotypes.
"""

import os
import numpy as np
import pandas as pd
from bed_reader import open_bed


def load_plink(bed_file, bim_file=None, fam_file=None, pheno=None, 
               id_col_name=None, impute_method="mode", backingfile=None):
    """
    Load PLINK files into a NumPy array and optionally align phenotype.
    
    Parameters
    ----------
    bed_file : str
        Path to the .bed file.
    bim_file : str, optional
        Path to the .bim file. If None, inferred from bed_file.
    fam_file : str, optional
        Path to the .fam file. If None, inferred from bed_file.
    pheno : pandas.DataFrame, optional
        DataFrame of phenotypes (must contain id_col_name). Default None.
    id_col_name : str, optional
        Name of the column in pheno containing individual IDs.
    impute_method : str
        The imputation method. Options: "mode", "mean", "zero". Default "mode".
    backingfile : str, optional
        Path for backing file (not used in Python version, kept for compatibility).
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'snp_obj': dict with 'genotypes' (2D array), 'fam' (DataFrame), 
                     'bim' (DataFrame), 'bedfile' (path)
        - 'pheno': Phenotype data aligned to genotypes (if provided)
    """
    print("Starting PLINK load process...")
    
    if not os.path.exists(bed_file):
        raise FileNotFoundError(f".bed file not found: {bed_file}")
    
    # Infer bim and fam file paths if not provided
    if bim_file is None:
        bim_file = bed_file.replace('.bed', '.bim')
    if fam_file is None:
        fam_file = bed_file.replace('.bed', '.fam')
    
    if not os.path.exists(bim_file):
        raise FileNotFoundError(f".bim file not found: {bim_file}")
    if not os.path.exists(fam_file):
        raise FileNotFoundError(f".fam file not found: {fam_file}")
    
    # Read .bed file using bed_reader
    print("--> Reading .bed file...")
    bed = open_bed(bed_file)
    genotypes = bed.read()  # Read as float64, values: 0, 1, 2, nan
    
    # Read .fam file (sample information)
    fam = pd.read_csv(fam_file, sep='\s+', header=None,
                     names=['FID', 'IID', 'Father', 'Mother', 'Sex', 'Phenotype'])
    fam['sample.ID'] = fam['IID'].astype(str)
    
    # Read .bim file (SNP information)
    bim = pd.read_csv(bim_file, sep='\s+', header=None,
                     names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'])
    
    # Impute missing genotypes
    print(f"--> Imputing missing genotypes using method: '{impute_method}'...")
    if impute_method == "mode":
        # Impute by mode (most common value) per SNP
        for i in range(genotypes.shape[1]):
            col = genotypes[:, i]
            mask = ~np.isnan(col)
            if mask.sum() > 0:
                values, counts = np.unique(col[mask], return_counts=True)
                mode_value = values[np.argmax(counts)]
                col[~mask] = mode_value
                genotypes[:, i] = col
    elif impute_method == "mean":
        # Impute by mean per SNP
        for i in range(genotypes.shape[1]):
            col = genotypes[:, i]
            mask = ~np.isnan(col)
            if mask.sum() > 0:
                mean_value = np.nanmean(col)
                col[~mask] = mean_value
                genotypes[:, i] = col
    elif impute_method == "zero":
        # Impute with zero
        genotypes = np.nan_to_num(genotypes, nan=0.0)
    else:
        raise ValueError(f"Unknown imputation method: {impute_method}")
    
    # Create snp_obj dictionary
    snp_obj = {
        'genotypes': genotypes,
        'fam': fam,
        'bim': bim,
        'bedfile': bed_file
    }
    
    # Optional: align phenotypes if provided
    aligned_pheno = None
    if pheno is not None:
        if id_col_name is None:
            raise ValueError("id_col_name must be provided when pheno is supplied.")
        
        geno_ids = fam['sample.ID'].values
        print("--> Aligning phenotype to genotype IDs...")
        
        # Keep only individuals present in genotype
        aligned_pheno = pheno[pheno[id_col_name].isin(geno_ids)].copy()
        
        # Match order to genotype
        id_to_idx = {id_val: idx for idx, id_val in enumerate(geno_ids)}
        aligned_pheno['__temp_idx__'] = aligned_pheno[id_col_name].map(id_to_idx)
        aligned_pheno = aligned_pheno.sort_values('__temp_idx__').drop(columns='__temp_idx__')
        aligned_pheno = aligned_pheno.reset_index(drop=True)
        
        # Sanity check
        if not np.array_equal(aligned_pheno[id_col_name].values, geno_ids):
            print("Warning: Phenotype and genotype IDs are not perfectly aligned after matching.")
        else:
            print("Phenotype successfully aligned to genotype order.")
    
    print("--- PLINK Load Complete! ---")
    
    return {
        'snp_obj': snp_obj,
        'pheno': aligned_pheno
    }

