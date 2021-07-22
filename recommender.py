#Importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Reading CSV Files
ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

#Securing the original data 
ratings_unabridged = ratings

movies['genres'] = movies.genres.str.split('|')
ratings = pd.merge(movies, ratings)

all_genres = []
#Created a list of all genres
for i in range(len(movies.index)):
    for genre in movies['genres'][i]:
        if(genre not in all_genres):
            all_genres.append(genre)

all_genres.remove('(no genres listed)')

sums = [[0 for k in range(ratings['userId'].max())] for j in range(len(all_genres))]
cnt = [[0 for k in range(ratings['userId'].max())] for j in range(len(all_genres))]
    
for genre in range(len(all_genres)):
    for i in range(len(ratings.index)):
        if(all_genres[genre] in ratings['genres'][i]):
            sums[genre][ratings['userId'][i]-1] += ratings['rating'][i]
            cnt[genre][ratings['userId'][i] - 1] += 1

#Average rating of all genres
averages = [[0 for k in range(ratings['userId'].max())] for j in range(len(all_genres))]

for j in range(len(all_genres)):
    for k in range(ratings['userId'].max()):
        if(cnt[j][k] != 0):
            averages[j][k] = round(sums[j][k]/cnt[j][k], 2)
        else:
            averages[j][k] = 0

names = []

avgs = pd.DataFrame(averages)

count = 0
for i in range(ratings['userId'].max()):
    names.append("User" + str(i+1))

avgs.columns = names
avgs.set_index([pd.Index(all_genres)], inplace=True)
avgs = avgs.transpose()

# biased_data['num']=np.arange(len(biased_data))
# biased_data=biased_data[['num','Comedy','Sci-Fi']]
# biased_data.set_index('num',inplace=True)

# print(biased_data.head())
# print(biased_data)

# Making scatterplot to get a better understanding of the clusters formed
# plt.scatter(biased_data['Comedy'],biased_data['Sci-Fi'])
# plt.scatter(avgs[all_genres])
# plt.show()

# X=biased_data[['Comedy','Sci-Fi']]
# print(type(X))

""" Optimal Clusters - Atleast 5, as seen from the graph of num_clusters:
 the goal is to have low inertia & least number of clusters. One way is to use elbow method."""
# num_cluster=list(range(1,9))
# inertias=[]
# for i in num_cluster:
#     model=KMeans(n_clusters=i)
#     model.fit(X)
#     inertias.append(model.inertia_)
# plt.plot(num_cluster,inertias,'-o')
#plt.show()

# K Means Clustering Model Applied 
model=KMeans(n_clusters=19,random_state=1)
predictions=model.fit_predict(avgs)

# Data of each cluster is stored
clustered = pd.concat([avgs.reset_index(), pd.DataFrame({'group':predictions})], axis=1)

# plt.figure(figsize=(8,8))
# plt.scatter(clustered['Comedy'], clustered['Sci-Fi'], c=clustered['group'], s=20)


user_ratings = pd.merge(ratings_unabridged,movies[['movieId','title']],on='movieId')
user_pivot=pd.pivot_table(user_ratings, index='userId', columns='title', values='rating')

clustered = clustered.set_index('index')


clusters = clustered.groupby('group')
dict = {}
dict = clusters.groups

arr_clusters = []
# Cluster information is stored in a list
for i in range(19):
    arr_clusters.append(list(dict[i].values))

    


# Testing out the recommendations based on ratings by given by a random user
test_user = 18
test_movieID = 59784
test_rating = 4.0

cluster_avg = []

# Calculating averages of every cluster to identify which cluster the user behaves like the most
for i in range(len(arr_clusters)):
    sum = 0
    length = 0
    for j in arr_clusters[i]:
        user =  int(j[4:])
        abcd = ratings_unabridged[(ratings_unabridged['userId']==user) & (ratings_unabridged['movieId']==test_movieID)]
        if(abcd.empty == False):
            sum += int(abcd['rating'].values)
            length += 1
    if(length):
        cluster_avg.append(round(sum/length,2))
    else:
        cluster_avg.append(0)



diff = 5
for i in range(len(cluster_avg)):
    if(abs(test_rating - cluster_avg[i]) < diff):
        optimal_cluster_index = i
        diff = abs(test_rating - cluster_avg[i])


df_list = list()

for i in arr_clusters[optimal_cluster_index]:
    user = int(i[4:])
    df_list.append(ratings_unabridged[(ratings_unabridged['userId'] == user)])


optimal_cluster_data = pd.DataFrame()

for i in df_list:
    optimal_cluster_data = optimal_cluster_data.append(i)

# Filtering data on the basis of ratings and number of users
optimal_cluster_data['counts'] = optimal_cluster_data.groupby(['movieId'])['userId'].transform('count')

optimal_cluster_data = optimal_cluster_data.groupby('movieId').mean().sort_values(by=['counts', 'rating'], ascending=False)

result_movie_ids = list(optimal_cluster_data.head().index)

#Final Result
for i in result_movie_ids:
    print(list(movies['title'][movies['movieId'] == i])[0])