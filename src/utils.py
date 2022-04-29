
from scipy import stats
from scipy.stats import norm, skew
import numpy as np
import pandas as pd

def bool_to_vec(df, cols):

    for col in cols:

        if df[col].dtypes == bool:

            df.loc[:,col] = df[col].apply(lambda x: 1 if x else 0)

    return df

def missing_values_replacement(df, cols_list):

    for col in cols_list:
        df.loc[:,col] =df[col].fillna(0)

    return df


def set_categorical_features(df):

    for c_idx, c in enumerate(df.columns):
        col_type = df[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            df[c] = df[c].astype('category')

    return df


def outlier_removal(df:pd.DataFrame, col):
    """
    input: dataframe
    purpose: removes outliers > 4 STD
    output: dataframe
    """
    z_scores = stats.zscore(df[{col}])
    abs_z_scores = np.abs(z_scores)

    filtered_entries = (abs_z_scores < 4).all(axis=1)
    outliers=(abs_z_scores >= 4).all(axis=1)
    df = df[filtered_entries]
    return df

def log_transformation(df:pd.DataFrame, cols_list):
    """
    input: dataframe
    purpose: log transformation
    output: dataframe
    """
    for col in cols_list:
        df.loc[:,col] = df[col].apply(lambda x: np.log1p(x))

    return df
