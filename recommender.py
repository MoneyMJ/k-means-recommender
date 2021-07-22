import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")

ratings_unabridged = ratings


# print(ratings.columns)

# userRatings = ratings.pivot_table(index = ['userId'], columns = ['title'], values = 'rating')
# print(userRatings.head())
#print(movies.head())

movies['genres'] = movies.genres.str.split('|')
ratings = pd.merge(movies, ratings)
#print(movies.head())


print(ratings_unabridged)

all_genres = []

for i in range(len(movies.index)):
    for genre in movies['genres'][i]:
        if(genre not in all_genres):
            all_genres.append(genre)

all_genres.remove('(no genres listed)')
# print(all_genres)

# print(ratings.head())

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

    
# print(sums)
# print (cnt)

averages = [[0 for k in range(ratings['userId'].max())] for j in range(len(all_genres))]

for j in range(len(all_genres)):
    for k in range(ratings['userId'].max()):
        if(cnt[j][k] != 0):
            averages[j][k] = round(sums[j][k]/cnt[j][k], 2)
        else:
            averages[j][k] = 0

# print(averages)

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

# print(avgs.head())

def bias_data(avgs, cutoff):
    new_data = avgs[(avgs['Comedy'] > cutoff) | (avgs['Sci-Fi'] > cutoff)]

    return new_data 


biased_data = bias_data(avgs, 3.5)

all_data = biased_data

biased_data = biased_data[['Comedy', 'Sci-Fi']]
# biased_data['num']=np.arange(len(biased_data))
# biased_data=biased_data[['num','Comedy','Sci-Fi']]
# biased_data.set_index('num',inplace=True)

# print(biased_data.head())
# print(biased_data)

# plt.scatter(biased_data['Comedy'],biased_data['Sci-Fi'])
# plt.scatter(avgs[all_genres])
plt.xlabel('Komedi')
plt.ylabel('Sci-Fi')
plt.title('Scatter')
# plt.show()

# X=biased_data[['Comedy','Sci-Fi']]
X = avgs
# print(type(X))

""" Optimal Clusters : Atleast 5"""
# num_cluster=list(range(1,9))
# inertias=[]
# for i in num_cluster:
#     model=KMeans(n_clusters=i)
#     model.fit(X)
#     inertias.append(model.inertia_)
# plt.plot(num_cluster,inertias,'-o')
#plt.show()

model=KMeans(n_clusters=19,random_state=1)
predictions=model.fit_predict(X)
print(len(predictions))


# def draw_clusters(biased_dataset, predictions, cmap='viridis'):
#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_subplot(1,1,1)
#     plt.xlim(0, 5)
#     plt.ylim(0, 5)
#     ax.set_xlabel('Avg scifi rating')
#     ax.set_ylabel('Avg romance rating')

clustered = pd.concat([avgs.reset_index(), pd.DataFrame({'group':predictions})], axis=1)

plt.figure(figsize=(8,8))
plt.scatter(clustered['Comedy'], clustered['Sci-Fi'], c=clustered['group'], s=20)


# print(X.columns)
# draw_clusters(biased_data,predictions)
# print(predictions)
# plt.show()

user_ratings = pd.merge(ratings_unabridged,movies[['movieId','title']],on='movieId')
user_pivot=pd.pivot_table(user_ratings, index='userId', columns='title', values='rating')
# print(user_pivot.head())

# Define Function to get the most rated movies
# list=[]
# for i in range(len(20)):

clustered = clustered.set_index('index')
print(clustered)

clusters = clustered.groupby('group')
dict = {}
dict = clusters.groups
# print(dict)
arr_clusters = []

# print(type(dict[0]))
for i in range(19):
    
    arr_clusters.append(list(dict[i].values))

# print(arr_clusters[0])

# print(type(arr_clusters[0][0]))

# for i in range(0, clusters.ngroups):
#     col = []
#     for j in 
#     col.append(clusters.get_group(str(i)))
    
print(arr_clusters)
# print(clusters.get_group(1))
# for i in range(0,19):
#     arr_clusters[i] = clusters.get_group(i)['index']

# print(arr_clusters.head())

# print(dict)

# print(avgs)

test_user = 18
test_movieID = 59784
test_rating = 4.0

cluster_avg = []
# print(type(ratings_unabridged['rating'][(ratings_unabridged['userId']==18) & (ratings_unabridged['movieId']==test_movieID)]))

for i in range(len(arr_clusters)):
    sum = 0
    length = 0
    for j in arr_clusters[i]:
        user =  int(j[4:])
        # print(user)
        abcd = ratings_unabridged[(ratings_unabridged['userId']==user) & (ratings_unabridged['movieId']==test_movieID)]
        if(abcd.empty == False):
            sum += int(abcd['rating'].values)
            length += 1
    if(length):
        cluster_avg.append(round(sum/length,2))
    else:
        cluster_avg.append(0)

print(cluster_avg)

diff = 5
for i in range(len(cluster_avg)):
    if(abs(test_rating - cluster_avg[i]) < diff):
        optimal_cluster_index = i
        diff = abs(test_rating - cluster_avg[i])

print(optimal_cluster_index)

DF_List = list()

for i in arr_clusters[optimal_cluster_index]:
    user = int(i[4:])
    DF_List.append(ratings_unabridged[(ratings_unabridged['userId'] == user)])

print(DF_List)

optimal_cluster_data = pd.DataFrame()

for i in DF_List:
    optimal_cluster_data = optimal_cluster_data.append(i)

# print(optimal_cluster_data)



# optimal_cluster_data = optimal_cluster_data[optimal_cluster_data.groupby('movieId').userId.count() > 10]
optimal_cluster_data['counts'] = optimal_cluster_data.groupby(['movieId'])['userId'].transform('count')
print(optimal_cluster_data['counts'].max())
optimal_cluster_data = optimal_cluster_data.groupby('movieId').mean().sort_values(by=['counts', 'rating'], ascending=False)

# optimal_cluster_data = optimal_cluster_data[optimal_cluster_data.mean().sort_values(by=['rating'], ascending=False)]
print(optimal_cluster_data.head())

result_movie_ids = list(optimal_cluster_data.head().index)
# print(result_movie_ids)



#Final Result
for i in result_movie_ids:
    print(list(movies['title'][movies['movieId'] == i])[0])