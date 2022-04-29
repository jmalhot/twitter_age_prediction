import json
import pandas as pd
import pickle
import os
import pandas as pd
from configs import DATAFILES
from proprocessing import preprocessing
from utils import *
from eda import eda
from model import model_training, model_prediction


def load_csv_data(path):
    """
    input: csv path
    purpose: load
    output: dataframe
    """

    return pd.read_csv(path)


def load_url_data(profile_url, profile_path, cache_file=True):
    """
    input: urls
    purpose: load
    output: dataframe
    """

    cache_path =str(profile_path) + ".cache.pickle"

    if cache_file and os.path.exists(cache_path):
        print("Loading user_profile from cache: " + cache_path)
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
            data = cache
    else:
        data = pd.read_json(profile_url)
        if cache_file:
            print("Dumping cache: " + cache_path)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return data


def load_dataset():
    """
    input: None
    purpose: load datasets
    output: dataframe
    """

    df_ages_train = load_csv_data(DATAFILES.AGES_TRAIN)
    df_ages_test = load_csv_data(DATAFILES.AGES_TEST)
    df_ages_test.columns= ['ID']
    df_mentions = load_csv_data(DATAFILES.MENTIONS )
    df_friends = load_csv_data(DATAFILES.FRIENDS)

    df_mentions_count = df_mentions.ID.value_counts().to_frame().reset_index()
    df_mentions_count.columns=['ID', 'mentions_count']


    df_user_profile = load_url_data(DATAFILES.USER_PROFILE_URL, DATAFILES.ROOT/'data/user')
    df_tweets = load_url_data(DATAFILES.TWEETS_URL, DATAFILES.ROOT/'data/tweets')
    df_friends_profile = load_url_data(DATAFILES.FRIENDS_PROFILE_URL, DATAFILES.ROOT/'data/friends')
    df_mention_profile =load_url_data(DATAFILES.MENTION_PROFILE_URL, DATAFILES.ROOT/'data/mention')



    df_user_profile, df_tweets ,_ , _ = preprocessing(df_user_profile, df_tweets, df_friends_profile, df_mention_profile)

    df_user_profile = pd.merge(df_user_profile, df_mentions_count, left_on=['id'], right_on=['ID'], how='left')

    df = pd.merge(df_user_profile, df_tweets, left_on=['id'], right_on=['user_id'], how='left')

    df = pd.merge(df, df_ages_train, right_on='ID', left_on='user_id')

    df.drop(columns=['ID_x','ID_y'], inplace=True)

    df= set_categorical_features(df)

    eda(df)

    '''
    Split Train and Test here
    '''

    df_test  = pd.merge(df, df_ages_test, left_on=['user_id'], right_on =['ID'])
    df['included_in_test_set'] = df['user_id'].apply(lambda x: 1 if x in df_ages_test.ID.unique().tolist() else 0)
    df_train = df[df.included_in_test_set ==0]

    df_train.drop(columns=['included_in_test_set'], inplace=True)


    df_train = outlier_removal(df_train, ['Age','friends_count','listed_count'])

    df_train = log_transformation(df_train, ['followers_count',
                                'listed_count',
                                'friends_count',
                                'retweet_count',
                                'tweets_retweet_count',
                                'tweets_favorite_count',
                                'tweet_length',
                                'emojis_count'
                                ]
                            )

    df_test = log_transformation(df_test, ['followers_count',
                                'listed_count',
                                'friends_count',
                                'retweet_count',
                                'tweets_retweet_count',
                                'tweets_favorite_count',
                                'tweet_length',
                                'emojis_count'
                                ]
                            )
    df_train = bool_to_vec(df_train, ['geo_enabled','default_profile','profile_background_tile'])
    df_test = bool_to_vec(df_test, ['geo_enabled','default_profile','profile_background_tile'])

    df_train = missing_values_replacement(df_train, ['friends_count', 'retweet_count', 'mentions_count'])

    df_test = missing_values_replacement(df_test, ['friends_count', 'retweet_count', 'mentions_count'])

    df_train.loc[:,'location'] = df_train['location'].apply(lambda x: 'unknown' if x=='' else x)

    model_training(df_train)

    model_prediction(df_test)


def main():
    """
    input: None
    purpose: Main function
    output: None
    """

    try:

        l_debug = ' Main Started: '
        load_dataset()
        l_debug = ' Main Ended: '

        print(l_debug)


    except Exception as e:
        l_debug = "Exception at  "+ l_debug + ' stage - ' +str(e)
        raise ValueError(l_debug)



if __name__ == '__main__':

    main()
