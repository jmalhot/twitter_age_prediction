"""
@overview: model training and evaluation
@author: J Malhotra
#@Created: April 2022
"""

from datetime import datetime
import category_encoders as ce
from sklearn.pipeline import Pipeline
#from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pickle
import pandas as pd
import numpy as np
from utils import CONSTANTS
from utils import compute_metrics


def save_validations_result(df):
    """
    input: dataframe
    purpose: Saves model prediction results
    output: None
    """

    df.to_csv(CONSTANTS.MODEL_PREDICTIONS_OUTPUT_PATH, index=False)



def drop_user_id_col(df):
    """
    input: dataframe
    purpose: drops user_id feature
    output: dataframe
    """

    df= df.drop(columns = ['user_id'])

    return df



def model_load():

    """
    input: None
    purpose: loads trained model for prediction
    output: model
    """


    l_debug = ' - Model loading from : {}'.format(CONSTANTS.MODEL_PATH)
    print(l_debug)
    model = pickle.load( open(CONSTANTS.MODEL_PATH, "rb" ) )

    return model

def model_dump(model):
    """
    input: trained model
    purpose: dumps model as a pickle
    output: None
    """
    try:

        l_debug = ' - Model Saving at Path: {}'.format(CONSTANTS.MODEL_PATH)
        print(l_debug)
        pickle.dump(model, open(CONSTANTS.MODEL_PATH,'wb'))

    except Exception as e:
        l_debug = "Exception at  "+ l_debug + ' stage - ' +str(e)
        raise ValueError(l_debug)


def model_prediction(test_df: pd.DataFrame):
    """
    input: dataframe
    purpose: model performance evaluation
    output: None
    """


    l_debug = 'Model Predicted started......'
    print(l_debug)

    model = model_load()

    #CONSTANTS.INPUT_FEATURES.pop(13)

    X_test=test_df[CONSTANTS.INPUT_FEATURES]
    y_test=test_df[CONSTANTS.OUTPUT_FEATURES]

    print(X_test.columns)

    l_debug = ' - Predict method called......'
    print(l_debug)
    y_pred=model.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns=CONSTANTS.OUTPUT_FEATURES

    for output_feature in CONSTANTS.OUTPUT_FEATURES:
        X_test[output_feature] = y_test[output_feature]
        X_test['pred_'+output_feature] = y_pred_df[output_feature]

    '''
    Convert log transformed feature to original values
    '''
    for output_feature in ['Age']:
        X_test[output_feature] = X_test[output_feature].apply(lambda x: np.expm1(x))

    save_validations_result(X_test)

    compute_metrics(y_test.to_numpy(), y_pred)


def model_training(df: pd.DataFrame):
    """
    input: dataframe
    purpose: Model Training using sklearn pipeline
    output: None
    """

    l_debug = 'Model Training started......'
    print(l_debug)

    CONSTANTS.INPUT_FEATURES.pop(14)

    X_train=df[CONSTANTS.INPUT_FEATURES]
    y_train=df[CONSTANTS.OUTPUT_FEATURES]

    print(X_train.columns)

    cat_vars=['creation_day_of_week']

    l_debug = ' - Model training pipeline creation......'
    print(l_debug)

    hyper_params = CONSTANTS.HYPER_PARAMS

    pipeline=Pipeline([
                ('target',ce.TargetEncoder(cols=cat_vars, handle_missing='value',handle_unknown='value',smoothing=5)),
                #('training',xgb.XGBRegressor(random_state=0))
                 ('training',LGBMRegressor(**hyper_params))
                 ]
              )

    start_time = datetime.now()
    model_reg=pipeline.fit(X_train,y_train)
    print(' - Model Training Time: {}'.format(datetime.now() - start_time))



    model_dump(model_reg)

def model_tunning(model):
    '''
    Placeholder for HyperParameter Tunning using Random Search/Grid Search
    '''
    pass
