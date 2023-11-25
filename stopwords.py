#import libraries
import pandas as pd
from nltk.corpus import stopwords
import re

stops = set(stopwords.words('turkish'))


#load preprocessed csv file
df = pd.read_csv('tweetler.csv')

tweets = df['Tweet'].tolist()

#remove stopwords
new_sent = []
for tweet in df['Tweet']:
    words = re.findall(r'\b\w+\b', tweet)
    new_sentence = [w for w in words if w.lower() not in stops]
    new_sent.append(' '.join(new_sentence))

df['Tweet'] = new_sent

df.to_csv('cleaned_tweets.csv', index=False)
