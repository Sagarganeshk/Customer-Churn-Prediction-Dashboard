import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and encodes the churn dataset.
    """
    # Encode categorical features
    for col in df.select_dtypes(include='object').columns:
        if col != 'customerID':
            df[col] = LabelEncoder().fit_transform(df[col])
    
    # Drop customerID (not useful for model)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    return df
