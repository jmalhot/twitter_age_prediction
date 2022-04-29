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
