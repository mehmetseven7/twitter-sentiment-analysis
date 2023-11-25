#import libraries
import emoji
import pandas as pd
import re

#read csv file for preprocessing
#df = pd.read_csv('csvfile.csv')

#testString = "bu. test? Stringidir, :smile: @kullanıcı #asdasdasd https://www.google.com"

def demojize_emojis(tweet):
    return emoji.demojize(tweet)

def remove_punctuation(tweet):
    #This regular expression line matches all characters that is not a white space or word character
    punctuation_pattern = re.compile(r'[^\w\s]')
    #remove punctuations by changing empty string
    clean_text = re.sub(punctuation_pattern, '', tweet)
    return clean_text

def preprocess_tweet(tweet):
    #convert tweets to lower case
    tweet = tweet.lower()

    #Remove URL's in tweets
    tweet = re.sub(r'((www\.[\S]+) | (https?://[\S]+))', '', tweet)

    #demojize emojis and remove demozjized strings such as :smile:
    tweet = demojize_emojis(tweet)
    tweet = re.sub(r':[a-z_]+:', '', tweet)

    #remove @mentions
    tweet = re.sub(r'@[\S]+', '', tweet)

    #remove # hashtags
    tweet = re.sub(r'#[\S]+', '', tweet)

    #remove RT retweets
    tweet = re.sub(r'\brt\b', '', tweet)

    #remove multiple dots...
    tweet = re.sub(r'\.{2,}', ' ', tweet)

    #Strip space " and ' from tweet
    tweet = tweet.strip(' "\'')

    #replace multiple spaces with single space
    tweet = re.sub(r'\s+', ' ', tweet)

    #remove punctuations
    tweet = remove_punctuation(tweet)

    return tweet


#read csv file for preprocessing
df = pd.read_csv('new_test.csv')

#apply preprocess to csv file
df['text'] = df['text'].apply(preprocess_tweet)

#save reults to new csv file
df.to_csv('preprocessed_tweets.csv', index=False)