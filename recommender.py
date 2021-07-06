import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")



#print(ratings.columns)

# userRatings = ratings.pivot_table(index = ['userId'], columns = ['title'], values = 'rating')
# print(userRatings.head())
#print(movies.head())

movies['genres'] = movies.genres.str.split('|')
ratings = pd.merge(movies, ratings)
print(movies.head())

all_genres = []

for i in range(len(movies.index)):
    for genre in movies['genres'][i]:
        if(genre not in all_genres):
            all_genres.append(genre)

all_genres.remove('(no genres listed)')
print(all_genres)

print(ratings.head())

# for genre in movies['genres']:
#     if(genre is in all_genres):
#         pass
#     else:
#         all_genres.append(genre)

# sum=0
# for user in range(1, ratings['userId'].nunique() + 1):
#     for genre in all_genres:
#         if user == ratings['userId']:



sums = []
for genre in range(len(all_genres)):
    for user in range(1, ratings['userId'].max() + 1):
        for i in range(len(ratings.index)):
            if[user == ratings['userId'][i] and all_genres[genre] in ratings['genres'][i]]:
                sums[genre] += ratings['rating'][i]
    
    
print(sums)