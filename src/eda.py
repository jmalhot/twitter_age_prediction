
"""
@overview: EDA/Visualization
@author: J Malhotra
#@Created: April 2022
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import stats
from scipy.stats import norm, skew

from pathlib import Path
ROOT_DIR = Path(__file__).parents[0].absolute()
sys.path.append(ROOT_DIR)
from configs import CONSTANTS


def error_plot(actual, predicted):
    """
    input: validation set true pain scores and predicted pain scores
    purpose: scatter plot to show true vs predicted points
    output: None
    """

    ig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(actual, predicted)
    ax.plot([actual.min(), actual.max()+1], [actual.min(), actual.max()+1], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Error Plot of Actual vs Predicted Pain Score")
    ax.legend(['Predicted', 'True'])

    plt.show()
    ig.savefig(CONSTANTS.IMAGES_OUTPUT_PATH / 'error_plot.jpg', dpi=ig.dpi)

def count_plot(df:pd.DataFrame, col1:str, col2:str, fig_name:str = 'countplot.jpg'):
    """
    input: dataframe and features
    purpose: count plot
    output: None
    """

    ig = plt.figure(figsize=(10,10))
    sns.set_theme(style="darkgrid")

    ax = sns.countplot(x=col1, hue=col2, data=df)
    plt.show()
    ig.savefig(CONSTANTS.IMAGES_OUTPUT_PATH / 'count_plot.jpg', dpi=ig.dpi)

def correlation_matrix(df:pd.DataFrame):
    """
    input: dataframe and features
    purpose: computes correlation between different features
    output: None
    """

    ig = plt.figure(figsize=(15,15))
    sns.heatmap(df.corr('spearman'), square=True, annot=True, cmap='Reds',linecolor="black", linewidths=0.20)
    plt.show()
    ig.savefig(CONSTANTS.IMAGES_OUTPUT_PATH / 'correlation_matrix.jpg', dpi=ig.dpi)


def distribution_plot(df:pd.DataFrame, col:str):
    """
    input: dataframe and features
    purpose: data distribution plot
    output: None
    """
    ig = plt.figure(figsize=(15,10))
    plt.subplot(1,2,1)
    sns.distplot(df[col] , fit=norm);
    plt.ylabel('Frequency')
    plt.title(col + ' - Distribution')

    plt.subplot(1,2,2)
    stats.probplot(df[col], plot=plt)
    plt.show()

    ig.savefig(CONSTANTS.IMAGES_OUTPUT_PATH / str('distribution_plot-'+col+'.jpg'), dpi=ig.dpi)


def bar_plot(df:pd.DataFrame, col:str):
    """
    input: dataframe
    purpose: bar plot
    output: None
    """

    df[col].value_counts().plot(kind='barh', figsize=(10,4), color='b')
    ig.savefig(CONSTANTS.IMAGES_OUTPUT_PATH / str('bar_plot-'+col+'.jpg'), dpi=ig.dpi)


def eda(df:list):
    """
    input: dataframe
    purpose: initiate eda and visualizations
    output: None
    """

    distribution_plot(df, 'Age')
    distribution_plot(df, 'followers_count')
    distribution_plot(df, 'listed_count')
    distribution_plot(df, 'friends_count')

    correlation_matrix(df)

    count_plot(df, 'creation_day_of_week' , 'positive_sentiment')
