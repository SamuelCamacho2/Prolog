import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "Popocatépetl"
tweets = []
limit = 50

for tweet in sntwitter.TwitterCashtagScraper(query).get_items():
    
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.content])

df = pd.DataFrame(tweets, columns=['Date','User','Tweet'])
print(df)

df.to_csv('Tweets2.csv')