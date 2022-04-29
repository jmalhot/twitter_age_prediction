



import json
import pandas as pd
import pickle
import os
import pandas as pd
from configs import DATAFILES
from proprocessing import preprocessing
from utils import *

def load_csv_data(path):

    return pd.read_csv(path)


def load_url_data(profile_url, profile_path, cache_file=True):

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

    df_ages_train = load_csv_data(DATAFILES.AGES_TRAIN)
    df_ages_test = load_csv_data(DATAFILES.AGES_TEST)
    df_mentions = load_csv_data(DATAFILES.MENTIONS )
    df_friends = load_csv_data(DATAFILES.FRIENDS)

    df_mentions_count = df_mentions.ID.value_counts().to_frame().reset_index()
    df_mentions_count.columns=['ID', 'mentions_count']


    df_user_profile = load_url_data(DATAFILES.USER_PROFILE_URL, DATAFILES.ROOT/'data/user')
    df_tweets = load_url_data(DATAFILES.TWEETS_URL, DATAFILES.ROOT/'data/tweets')
    df_friends_profile = load_url_data(DATAFILES.FRIENDS_PROFILE_URL, DATAFILES.ROOT/'data/friends')
    df_mention_profile =load_url_data(DATAFILES.MENTION_PROFILE_URL, DATAFILES.ROOT/'data/mention')



    preprocessing(df_user_profile, df_tweets, df_friends_profile, df_mention_profile)

    df_user_profile = pd.merge(df_user_profile, df_mentions_count, left_on=['id'], right_on=['ID'], how='left')

    df = pd.merge(df_user_profile, df_tweets, left_on=['id'], right_on=['user_id'], how='left')

    df = pd.merge(df, df_ages_train, right_on='ID', left_on='user_id')

    #df.drop(columns=['id','ID_x'], inplace=True)

    #df= set_categorical_features(df)

    df = outlier_removal(df, ['Age','friends_count','listed_count'])

    df = log_transformation(df, ['followers_count',
                                'listed_count',
                                'friends_count',
                                'retweet_count',
                                'tweets_retweet_count',
                                'tweets_favorite_count',
                                'tweet_length',
                                'emojis_count'
                                ]
                            )
    df = bool_to_vec(df, ['geo_enabled','default_profile','profile_background_tile'])



    df = missing_values_replacement(df, ['friends_count', 'retweet_count', 'mentions_count'])



    #df.loc[:,'friends_count'] =df['friends_count'].fillna(0)
    #df.loc[:,'retweet_count'] =df['retweet_count'].fillna(0)
    #df.loc[:,'mentions_count'] =df['retweet_count'].fillna(0)
    df.loc[:,'location'] = df['location'].apply(lambda x: 'unknown' if x=='' else x)



    #df_user_profile.to_csv('df_user_profile.csv')
    #df_tweets.to_csv('df_tweets.csv')
    df.to_csv("df.csv")



def main():

    try:

        l_debug = ' Main Started: '
        load_dataset()

    except Exception as e:
        l_debug = "Exception at  "+ l_debug + ' stage - ' +str(e)
        raise ValueError(l_debug)



if __name__ == '__main__':

    main()
