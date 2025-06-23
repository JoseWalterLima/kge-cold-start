# Author: José Walter Mota
# 02/2025

import os
import pandas as pd
directory = "data/"
os.makedirs(directory, exist_ok=True)

# Movie dataset
m=pd.read_csv(
    'data/ml-100k/u.item',
    sep='|',
    encoding="latin1",
    names=[
        'movieId', 'movieTitle', 'releaseDate', 'videoReleaseDate',
        'IMDbURL', 'unknown', 'Action', 'Adventure', 'Animation',
        'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
        'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
           ).drop(['unknown', 'videoReleaseDate', 'IMDbURL'], axis=1)
# Reshaping
m=m.melt(
    id_vars=["movieId", "movieTitle", "releaseDate"],
    var_name="genreDesc",
    value_name="is_genre"
    ).copy()
m=m[m["is_genre"]==1].drop("is_genre", axis=1)\
    .reset_index(drop=True).copy()
# Extract features
m['movieTitle']=m['movieTitle'].str[:-7]
m['releaseDate']=m['releaseDate'].str[3:]

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
# Keep only valid movies
w=pd.read_csv('data/ml-100k/u.data', sep='\t',
names=['userId', 'movieId', 'rating', 'timestamp'])[['userId', 'movieId']]
w=w[w['movieId'].isin(m['movieId'].unique().tolist())]
w.to_csv('data/watchedRel.csv', index=False)

# User dataset
u=pd.read_csv('data/ml-100k/u.user', sep='|',
names=['userId', 'age', 'gender', 'occupation', 'zipcode'])
# simplify location
u['zipcode']=u['zipcode'].str[:-3]

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
