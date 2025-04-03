import pandas as pd
import os
import logging

def load_data(file_paths, column_mapping):
    """
    Load CSV, JSON, and Excel files from provided file paths,
    normalize column names based on the mapping, and return a unified DataFrame.
    """
    dfs = []
    for file in file_paths:
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.endswith('.json'):
                df = pd.read_json(file)
            elif file.endswith('.xls') or file.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                logging.warning(f"Unsupported file format: {file}")
                continue
            df = normalize_columns(df, column_mapping)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error loading file {file}: {e}")
    if dfs:
        data = pd.concat(dfs, ignore_index=True)
        return data
    else:
        return pd.DataFrame()

def normalize_columns(df, column_mapping):
    """
    Rename columns in df to standard names according to the mapping dictionary.
    The mapping dictionary should be structured as:
    { canonical_name: [list, of, variant, names] }
    """
    for canonical, variants in column_mapping.items():
        for variant in variants:
            if variant in df.columns:
                df.rename(columns={variant: canonical}, inplace=True)
    return df
