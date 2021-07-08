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
#print(movies.head())

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



# sums = []
# for j in range(len(all_genres)):
#     for k in range(ratings['userId'].max()):
#         sums.append(0)

sums = [[0 for k in range(ratings['userId'].max())] for j in range(len(all_genres))]
cnt = [[0 for k in range(ratings['userId'].max())] for j in range(len(all_genres))]
    
# for genre in range(len(all_genres)):
#     for user in range(1, ratings['userId'].max() + 1):
#         for i in range(len(ratings.index)):
#             if[user == ratings['userId'][i] and all_genres[genre] in ratings['genres'][i]]:
#                 sums[genre] += ratings['rating'][i]
    
for genre in range(len(all_genres)):
    for i in range(len(ratings.index)):
        if(all_genres[genre] in ratings['genres'][i]):
            sums[genre][ratings['userId'][i]-1] += ratings['rating'][i]
            cnt[genre][ratings['userId'][i] - 1] += 1

    
print(sums)
print (cnt)

averages = [[0 for k in range(ratings['userId'].max())] for j in range(len(all_genres))]

for j in range(len(all_genres)):
    for k in range(ratings['userId'].max()):
        if(cnt[j][k] != 0):
            averages[j][k] = round(sums[j][k]/cnt[j][k], 2)
        else:
            averages[j][k] = 0

print(averages)

names = []

avgs = pd.DataFrame(averages)
# avgs.transpose()
count = 0
for i in range(ratings['userId'].max()):
    names.append("User" + str(i+1))
    # avgs.columns[i] = "User" + str(i+1)
    #count += 1
    #avgs.columns[count] = ["User" +'{}'.format(count)]
avgs.columns = names
avgs.set_index([pd.Index(all_genres)], inplace=True)
avgs = avgs.transpose()

print(avgs.head())
