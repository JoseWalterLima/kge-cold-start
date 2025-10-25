# Author: José Walter Mota
# 02/2025
"""
Need to add a description here!
"""
import os
import requests
import zipfile
import io
import pandas as pd
data_url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
directory = "data/"
os.makedirs(directory, exist_ok=True)

# Download the data from Movielens official website
def download_and_extract_movielens():
    url = data_url
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=directory)
download_and_extract_movielens()
# Build Movies dataset
m=pd.read_csv(
    'data/ml-1m/movies.dat',
    sep='::',
    encoding="latin1",
    names=[
        'movieId', 'movieTitle', 'Genres'],
        engine='python')
m['releaseDate']=m['movieTitle'].str[-5:-1]
m['movieTitle']=m['movieTitle'].str[:-7]
m['genreDesc'] = m['Genres'].str.split('|')
m = m.explode('genreDesc').reset_index(drop=True)
m = m.drop(columns=['Genres'])

# Movie Nodes
m[['movieId', 'movieTitle']].copy().drop_duplicates()\
    .dropna(how='all').reset_index(drop=True)\
    .to_csv('data/movieNode.csv', index=False)

m[['releaseDate']].copy().drop_duplicates()\
    .dropna(how='all').reset_index(drop=True)\
    .to_csv('data/releaseNode.csv', index=False)

m[['genreDesc']].copy().drop_duplicates()\
    .dropna(how='all').reset_index(drop=True)\
    .to_csv('data/genreNode.csv', index=False)

# Movie Relationships
m[['movieId', 'releaseDate']
         ].copy().drop_duplicates().dropna(how='all').reset_index(drop=True)\
            .to_csv('data/releaseRel.csv', index=False)

m[['movieId', 'genreDesc']
         ].copy().drop_duplicates().dropna(how='all').reset_index(drop=True)\
            .to_csv('data/genreRel.csv', index=False)

# Keep only valid movies in watched relationship
w=pd.read_csv('data/ml-1m/ratings.dat', sep='::',
names=['userId', 'movieId', 'rating', 'timestamp'],
engine='python')[['userId', 'movieId']]
w=w[w['movieId'].isin(m['movieId'].unique().tolist())]
w.to_csv('data/watchedRel.csv', index=False)

# User dataset
u=pd.read_csv('data/ml-1m/users.dat', sep='::',
names=['userId', 'gender', 'age', 'occupation', 'zipcode'])
# simplify location
u['zipcode']=u['zipcode'].str[:2]

# User Nodes
u[['userId']].copy().drop_duplicates()\
    .dropna(how='all').reset_index(drop=True)\
    .to_csv('data/userNode.csv', index=False)

u[['age']].copy().drop_duplicates()\
    .dropna(how='all').reset_index(drop=True)\
    .to_csv('data/ageNode.csv', index=False)

u[['gender']].copy().drop_duplicates()\
    .dropna(how='all').reset_index(drop=True)\
    .to_csv('data/genderNode.csv', index=False)

u[['occupation']].copy().drop_duplicates()\
    .dropna(how='all').reset_index(drop=True)\
    .to_csv('data/occupationNode.csv', index=False)

u[['zipcode']].copy().drop_duplicates()\
    .dropna(how='all').reset_index(drop=True)\
    .to_csv('data/zipcodeNode.csv', index=False)

# User Relationships
u[['userId', 'age']
         ].copy().drop_duplicates().dropna(how='all').reset_index(drop=True)\
            .to_csv('data/ageRel.csv', index=False)
u[['userId', 'gender']
         ].copy().drop_duplicates().dropna(how='all').reset_index(drop=True)\
            .to_csv('data/genderRel.csv', index=False)
u[['userId', 'occupation']
         ].copy().drop_duplicates().dropna(how='all').reset_index(drop=True)\
            .to_csv('data/occupationRel.csv', index=False)
u[['userId', 'zipcode']
         ].copy().drop_duplicates().dropna(how='all').reset_index(drop=True)\
            .to_csv('data/residesRel.csv', index=False)
