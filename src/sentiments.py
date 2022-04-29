import nltk
nltk.download('vader_lexicon')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import emoji
import regex

def evaluate_sentiments(p_txt:str):


    text_tb = TextBlob(p_txt)

    analyzer = SentimentIntensityAnalyzer()
    text_vs = analyzer.polarity_scores(p_txt)


    if text_tb.sentiment.polarity <= 0 and text_vs['compound'] <= -0.5:
        sentiment = "negative"  # very negative
    elif text_tb.sentiment.polarity <= 0 and text_vs['compound'] <= -0.1:
        sentiment = "negative"  # somewhat negative
    elif text_tb.sentiment.polarity == 0 and text_vs['compound'] > -0.1 and text_vs['compound'] < 0.1:
        sentiment = "neutral"
    elif text_tb.sentiment.polarity >= 0 and text_vs['compound'] >= 0.1:
        sentiment = "positive"  # somewhat positive
    elif text_tb.sentiment.polarity > 0 and text_vs['compound'] >= 0.1:
        sentiment = "positive"  # very positive
    else:
        sentiment = "neutral"

    polarity = (text_tb.sentiment.polarity + text_vs['compound']) / 2


    if  sentiment == 'negative':
        return 1, 0, 0
    elif sentiment == 'positive':
        return 0, 0, 1
    else:
        return 0, 1, 0


def count_emojis(text:str)-> int:

    e_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI['en'] for char in word):
            e_list.append(word)

    return len(e_list)
