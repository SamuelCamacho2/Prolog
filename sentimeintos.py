# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from scipy.special import softmax
import csv
tweet = 'Se fue el internet. 😞😞😞😞'

def leer_archivo_csv(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        lector_csv = csv.reader(archivo)
        for linea in lector_csv:
            # Procesar cada línea del archivo
            print(linea)  # Ejemplo: Imprimir la línea en la consola

# Ruta del archivo CSV
ruta_archivo_csv = 'Tweets.csv'

# Llamar a la función para leer el archivo CSV
leer_archivo_csv(ruta_archivo_csv)

# precprcess tweet
#tweet_words = []

# for word in tweet.split(' '):
#     if word.startswith('@') and len(word) > 1:
#         word = '@user'
    
#     elif word.startswith('http'):
#         word = "http"
#     tweet_words.append(word)

# tweet_proc = " ".join(tweet_words)

# # load model and tokenizer
# roberta = "cardiffnlp/twitter-roberta-base-sentiment"

# model = AutoModelForSequenceClassification.from_pretrained(roberta)
# tokenizer = AutoTokenizer.from_pretrained(roberta)

# labels = ['Negative', 'Neutral', 'Positive']

# # sentiment analysis
# encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
# output = model(**encoded_tweet)

# scores = output[0][0].detach().numpy()
# scores = softmax(scores)

# for i in range(len(scores)):
    
#     l = labels[i]
#     s = scores[i]
#     print(l,s)