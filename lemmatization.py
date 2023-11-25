import pandas as pd
from zeyrek import MorphAnalyzer

#Zeyrek lemmatizer
analyzer = MorphAnalyzer()

#read csv
df = pd.read_csv('cleaned_tweets.csv')

tweets = df['Tweet'].tolist()

#lemmatizing all tweets
lemmatized_tweets = []
for tweet in tweets:
    lemmatized_tweet = ' '.join([analyzer.lemmatize(word)[0][1][0] for word in tweet.split()])
    lemmatized_tweets.append(lemmatized_tweet)

df['Tweet'] = lemmatized_tweets

#save new results
df.to_csv('lemmatized_tweets.csv', index=False)