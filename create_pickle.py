import numpy as np
import pandas as pd
import ast
import pickle
import nltk

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits,on='title')

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.dropna(inplace=True)

def convert(objs):
    L=[]
    for obj in ast.literal_eval(objs):
        L.append(obj['name'])
    return L

movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)

def convert3(objs):
    L=[]
    cnt=0
    for obj in ast.literal_eval(objs):
        if cnt!=3:
            L.append(obj['name'])
            cnt+=1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(objs):
    L=[]
    for obj in ast.literal_eval(objs):
        if obj['job']=='Director':
            L.append(obj['name'])
            break
    return L

movies['crew']=movies['crew'].apply(fetch_director)

movies['overview']=movies['overview'].apply(lambda x:x.split())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df=movies[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

ps = PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

cv = CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

pickle.dump(new_df,open('movies.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

print("Pickle files created successfully")