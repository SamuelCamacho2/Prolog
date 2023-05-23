import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "Popocatépetl"
tweets = []
limit = 500

for tweet in sntwitter.TwitterCashtagScraper(query).get_items():
    
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.content])
    
    # print(vars(tweet))
    # break



df = pd.DataFrame(tweets, columns=['Date','User','Tweet'])
print(df)

df.to_csv('Tweets.csv')