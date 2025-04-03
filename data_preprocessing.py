import pandas as pd
from utils import load_data

def preprocess_data(file_paths):
    """
    Preprocess the manually added CSV/JSON files by:
    1. Defining a mapping for common parameter names.
    2. Loading and normalizing the data.
    3. Additional cleaning or feature engineering as needed.
    """
    # Mapping dictionary: adjust or extend as needed
    column_mapping = {
        "age": ["age", "Age", "patient_age"],
        "symptoms": ["symptoms", "Symptoms", "complaints"],
        "drug_name": ["drug_name", "DrugName", "medication"],
        "indication": ["indication", "Indication", "diagnosis"],
        "side_effects": ["side_effects", "SideEffects", "adverse_effects"],
        "interaction": ["interaction", "Interactions"],
        # Add more mappings for other parameters as necessary
    }
    data = load_data(file_paths, column_mapping)
    # Example cleaning: fill missing values
    data.fillna("", inplace=True)
    return data
