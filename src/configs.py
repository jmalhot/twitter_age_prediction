from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parents[0].absolute()
ROOT_DIR_PARENT = Path(__file__).parents[1].absolute()
sys.path.append(ROOT_DIR)

class DATAFILES:

    ROOT = ROOT_DIR_PARENT

    AGES_TRAIN = ROOT_DIR_PARENT / 'data/ages_train.csv'
    AGES_TEST  = ROOT_DIR_PARENT / 'data/ages_test.csv'
    FRIENDS  = ROOT_DIR_PARENT / 'data/friends.csv'
    MENTIONS  = ROOT_DIR_PARENT / 'data/mentions.csv'

    USER_PROFILE_URL = 'https://buapp-prod.s3.amazonaws.com/resources/hiring_ml_engineer_resources/age_profiles.json'
    FRIENDS_PROFILE_URL = 'https://buapp-prod.s3.amazonaws.com/resources/hiring_ml_engineer_resources/friend_profiles.json'
    MENTION_PROFILE_URL = 'https://buapp-prod.s3.amazonaws.com/resources/hiring_ml_engineer_resources/mention_profiles.json'
    TWEETS_URL = 'https://buapp-prod.s3.amazonaws.com/resources/hiring_ml_engineer_resources/age_tweets.json'

class CONSTANTS:
    USER_PROFILE_FEATURES =['id',
                             'geo_enabled',
                             'profile_sidebar_border_color',
                             'default_profile',
                             'followers_count',
                             'profile_sidebar_fill_color',
                             'listed_count',
                             'profile_background_color',
                             'profile_background_tile',
                             'profile_link_color',
                             'location',
                             'profile_text_color',
                             'friends_count',
                             'retweet_count']
    TWEET_FEATURES =[]

    OUTPUT_FEATURES = ['Age']

    INPUT_FEATURES = ['geo_enabled',
                    'profile_sidebar_border_color',
                    'default_profile',
                    'followers_count',
                    'profile_sidebar_fill_color',
                    'listed_count',
                    'profile_background_color',
                    'profile_background_tile',
                    'profile_link_color',
                    'location',
                    'profile_text_color',
                    'friends_count',
                    'retweet_count',
                    'mentions_count',
                    'user_id',
                    'lang',
                    'tweets_retweet_count',
                    'tweets_favorite_count',
                    'possibly_sensitive',
                    'source_parsed',
                    'creation_day_of_week',
                    'tweet_length',
                    'negative_sentiment',
                    'neutral_sentiment',
                    'positive_sentiment',
                    'emojis_count']

    HYPER_PARAMS = {'learning_rate': 0.1,
                   'boosting_type': 'gbdt',
                   'random_state':0,
                   'reg_alpha':1,
                   'reg_lambda':1
                  }

    MODEL_PATH = ROOT_DIR_PARENT / 'models/model.pkl'
    METRICS_PATH = ROOT_DIR_PARENT / 'models/metrics.txt'
    MODEL_PREDICTIONS_OUTPUT_PATH = ROOT_DIR_PARENT / 'data/ages_pred.csv'
    IMAGES_OUTPUT_PATH = ROOT_DIR_PARENT / 'visualization/'
