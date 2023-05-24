import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def graficar(negativo, neutal, positivo):
    etiquetas = ['negativo','neutal','positivo']
    valores = [negativo,neutal,positivo]
    
    co = np.arange(len(valores))
    an = 0.30
    
    fig, ax = plt.subplots()
    ax.bar(co, valores, an)
    
    ax.set_title('Grafica de emociones')
    ax.set_ylabel('Rank')
    ax.set_xticks(co)
    ax.set_xticklabels(etiquetas)

def archivo(busq,limite):
    query = busq
    tweets = []
    limit = limite
    for tweet in sntwitter.TwitterCashtagScraper(query).get_items():
        
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date, tweet.user.username, tweet.content])

    df = pd.DataFrame(tweets, columns=['Date','User','Tweet'])
    print(df)

    df.to_csv('Tweets2.csv')


def leer_archivo_csv(ruta_archivo):
    puntaje1 = 0
    puntaje2 = 0
    puntaje3 = 0
    
    negativo=0
    neutral=0
    positivo=0
    dividir =0
    with open(ruta_archivo, 'r') as archivo:
        lector_csv = csv.reader(archivo)
        next(archivo,None)   
        for linea in lector_csv:
            print('--------------------------------------------------') 
            print(linea[3]) 
            tweet = linea[3]


            # precprcess tweet
            tweet_words = []

            for word in tweet.split(' '):
                if word.startswith('@') and len(word) > 1:
                    word = '@user'
    
                elif word.startswith('http'):
                    word = "http"
                tweet_words.append(word)

            tweet_proc = " ".join(tweet_words)

            # load model and tokenizer
            roberta = "cardiffnlp/twitter-roberta-base-sentiment"

            model = AutoModelForSequenceClassification.from_pretrained(roberta)
            tokenizer = AutoTokenizer.from_pretrained(roberta)

            labels = ['Negative', 'Neutral', 'Positive']

            # sentiment analysis
            encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
            # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
            output = model(**encoded_tweet)

            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            puntaje1 = scores[0]
            puntaje2 = scores[1]
            puntaje3 = scores[2]
            dividir = dividir+1
            negativo = negativo + puntaje1
            neutral = neutral + puntaje2
            positivo = positivo + puntaje3
            for i in range(len(scores)):
                
                l = labels[i]
                s = scores[i]
                
                print(l,s)
            print()
            print(negativo, neutral, positivo)
    
    print(negativo/dividir, neutral/dividir, positivo/dividir)


graficar(10, 12, 5)

# archivo('Popocatépetl', 5)
# ruta_archivo_csv = 'Tweets2.csv'
# leer_archivo_csv(ruta_archivo_csv)