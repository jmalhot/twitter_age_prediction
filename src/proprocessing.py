
from bs4 import BeautifulSoup
from datetime import datetime
from sentiments import evaluate_sentiments, count_emojis
import pandas as pd
from configs import CONSTANTS

def parse_user_id(df:pd.DataFrame, new_col:str, target_col:str) -> pd.DataFrame:
    """
    input: dataframe and its column
    purpose: parse nested json field
    output: dataframe
    """

    df[new_col] = pd.json_normalize(df.user)[target_col]

    return df

def parse_retweet_count(df:pd.DataFrame, new_col:str, target_col:str) -> pd.DataFrame:
    """
    input: dataframe and its column
    purpose: parse nested json field
    output: dataframe
    """

    df[new_col] = pd.json_normalize(df.status)[target_col]

    return df




def parse_tweet_source(df:pd.DataFrame, new_col:str, target_col:str) -> pd.DataFrame:
    """
    input: dataframe and its column
    purpose: Parse Tweet Source using BeautifulSoup as it may have meaningfull information
    output: dataframe
    """

    df[new_col] = df[target_col].apply(lambda x: BeautifulSoup(str(x),'html.parser').text)

    return df

def get_day_of_week(date):
    """
    input: date
    purpose: extracts day of the week from date
    output: day of the week
    """
    l_day_of_week = date.strftime('%A')


    return l_day_of_week


def preprocessing(df_user_profile:pd.DataFrame,
                 df_tweets:pd.DataFrame,
                 df_friends_profile:pd.DataFrame,
                 df_mention_profile:pd.DataFrame):

    """
    input: dataframes
    purpose:feature engineering
    output: dataframes
    """


    df_tweets = preprocessing_tweets(df_tweets)
    df_user_profile = preprocessing_users(df_user_profile)
    '''
    Skipping for the purpose of this excercise
    df_friends_profile = preprocessing_tweets(df_friends_profile)
    df_mention_profile = preprocessing_tweets(df_mention_profile)
    '''

    return df_user_profile, df_tweets,df_friends_profile, df_mention_profile



def preprocessing_users(df_user_profile:pd.DataFrame):

    df_user_profile = parse_retweet_count(df_user_profile, 'retweet_count', 'retweet_count')



    return df_user_profile[CONSTANTS.USER_PROFILE_FEATURES]

def preprocessing_tweets(df_tweets:pd.DataFrame)->pd.DataFrame:

    df_tweets = parse_user_id(df_tweets, 'user_id', 'id')
    df_tweets = parse_tweet_source(df_tweets, 'source_parsed', 'source')

    df_tweets['creation_day_of_week'] = df_tweets['created_at'].apply(lambda x: get_day_of_week(x))

    df_tweets['tweet_length'] =df_tweets['text'].apply(lambda x: len(x))


    df_tweets['negative_sentiment'],df_tweets['neutral_sentiment'], df_tweets['positive_sentiment']  = zip(*df_tweets['text'].apply(lambda x: evaluate_sentiments(x)))

    df_tweets['emojis_count'] = df_tweets['text'].apply(lambda x: count_emojis(x))

    df_tweets['possibly_sensitive'] = df_tweets['possibly_sensitive'].fillna(-1)

    #df_tweets.drop(columns = ['created_at', 'source','entities'], inplace=True)

    df_tweets.rename(columns={'retweet_count':'tweets_retweet_count',
                         'favorite_count':'tweets_favorite_count'}, inplace=True)



    df_tweets = df_tweets.groupby('user_id',as_index=False)\
                                     .agg({
                                            'lang': lambda x: x.value_counts().index[0],
                                            'tweets_retweet_count':'mean',
                                            'tweets_favorite_count': 'mean',
                                            'possibly_sensitive': lambda x: x.value_counts().index[0],
                                            'source_parsed': lambda x: x.value_counts().index[0],
                                            'creation_day_of_week': lambda x: x.value_counts().index[0],
                                            'tweet_length': 'mean',
                                            'negative_sentiment': 'sum',
                                            'neutral_sentiment': 'sum',
                                            'positive_sentiment': 'sum',
                                            'emojis_count': 'sum',
                                          }
                                         )
    return df_tweets
