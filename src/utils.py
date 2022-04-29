
from scipy import stats
from scipy.stats import norm, skew
import numpy as np
import pandas as pd
from datetime import datetime
import os
from sklearn.metrics import mean_squared_log_error, r2_score, mean_squared_error
import numpy as np
from configs import CONSTANTS


def bool_to_vec(df, cols):
    """
    input: dataframe and its column
    purpose: bol to vector conversion
    output: dataframe
    """

    for col in cols:

        if df[col].dtypes == bool:

            df.loc[:,col] = df[col].apply(lambda x: 1 if x else 0)

    return df

def missing_values_replacement(df, cols_list):
    """
    input: dataframe and its columns
    purpose: replace missing values with 0
    output: dataframe
    """

    for col in cols_list:
        df.loc[:,col] =df[col].fillna(0)

    return df


def set_categorical_features(df):
    """
    input: dataframe
    purpose: set objects types to categorical
    output: dataframe
    """

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
    z_scores = stats.zscore(df[col])
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



def compute_metrics(actual , predicted):
    """
    input: validation set true pain scores and predicted pain scores
    purpose: computes metrics such as RMSE
    output: None
    """

    dt=datetime.now()

    l_debug = ' - Computing Metrics :'


    mse= round(mean_squared_error(actual, predicted),2)
    rmse=round(np.sqrt(mean_squared_error(actual, predicted)),2)
    r2= round(r2_score(actual, predicted),2)


    print(l_debug)
    print('     - Mean squared error: %.2f'% mse)
    print('     - Root Mean squared error: %.2f'% rmse)
    print('     - Coefficient of determination: %.2f'% r2)

    l_debug = ' - Saving Metrics to a file locally:'
    if not os.path.exists(CONSTANTS.METRICS_PATH):
        file= open(CONSTANTS.METRICS_PATH,'w')
        file.write('DATE,      MSE,  RMSE, R2')
        file.close()

    file = open(CONSTANTS.METRICS_PATH,'a')
    file.write('\n' +str(dt.strftime("%d-%m-%Y"))+','+str(mse)+','+str(rmse)+','+str(r2))
    file.close()

    error_plot(actual , predicted)
