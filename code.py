#Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
import base64
import io
from scipy.misc import imread
import codecs
from IPython.display import HTML
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
from scipy import spatial
import operator

#Import the data
movies=pd.read_csv('./input/tmdb_5000_movies.csv')
mov=pd.read_csv('./input/tmdb_5000_credits.csv')

#Preview Data
def preview():
    print(movies.head(1))
    print('\n\n\n\n')
    print(mov.head(1))

#Converting json into strings
def jsontostring():
    # changing the genres column from json to string
    movies['genres']=movies['genres'].apply(json.loads)
    for index,i in zip(movies.index,movies['genres']):
        list1=[]
        for j in range(len(i)):
            list1.append((i[j]['name']))# the key 'name' contains the name of the genre
        movies.loc[index,'genres']=str(list1)

    # changing the keywords column from json to string
    movies['keywords']=movies['keywords'].apply(json.loads)
    for index,i in zip(movies.index,movies['keywords']):
        list1=[]
        for j in range(len(i)):
            list1.append((i[j]['name']))
        movies.loc[index,'keywords']=str(list1)

    ## changing the production_companies column from json to string
    movies['production_companies']=movies['production_companies'].apply(json.loads)
    for index,i in zip(movies.index,movies['production_companies']):
        list1=[]
        for j in range(len(i)):
            list1.append((i[j]['name']))
        movies.loc[index,'production_companies']=str(list1)

    # changing the production_countries column from json to string
    movies['production_countries']=movies['production_countries'].apply(json.loads)
    for index,i in zip(movies.index,movies['production_countries']):
        list1=[]
        for j in range(len(i)):
            list1.append((i[j]['name']))
        movies.loc[index,'production_countries']=str(list1)

    # changing the cast column from json to string
    mov['cast']=mov['cast'].apply(json.loads)
    for index,i in zip(mov.index,mov['cast']):
        list1=[]
        for j in range(len(i)):
            list1.append((i[j]['name']))
        mov.loc[index,'cast']=str(list1)

    # changing the crew column from json to string
    mov['crew']=mov['crew'].apply(json.loads)
    def director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
    mov['crew']=mov['crew'].apply(director)
    mov.rename(columns={'crew':'director'},inplace=True)
    # print(movies.head(1))
    # print('\n\n\n\n')
    # print(mov.head(1))


def binary(genre_list):
    binaryList = []

    for genre in genreList:
        if genre in genre_list:
            binaryList.append(1)
        else:
            binaryList.append(0)

    return binaryList


def xstr(s):
    if s is None:
        return ''
    return str(s)

def Similarity(movieId1, movieId2):
    a = movies.iloc[movieId1]
    b = movies.iloc[movieId2]

    genresA = a['genres_bin']
    genresB = b['genres_bin']

    genreDistance = spatial.distance.cosine(genresA, genresB)
    scoreA = a['cast_bin']
    scoreB = b['cast_bin']
    scoreDistance = spatial.distance.cosine(scoreA, scoreB)
    directA = a['director_bin']
    directB = b['director_bin']
    directDistance = spatial.distance.cosine(directA, directB)
    wordsA = a['words_bin']
    wordsB = b['words_bin']
    wordsDistance = spatial.distance.cosine(directA, directB)
    resDistance = 0
    if(genreDistance==genreDistance):
        resDistance = resDistance+genreDistance
    if(scoreDistance==scoreDistance):
        resDistance = resDistance+scoreDistance
    if(directDistance==directDistance):
        resDistance = resDistance+directDistance
    if(wordsDistance==wordsDistance):
        resDistance = resDistance+wordsDistance
    return resDistance



def whats_my_score(name):
    print('Enter a movie title')
    new_movie=movies[movies['original_title'].str.contains(name)].iloc[0].to_frame().T
    print('Selected Movie: ',new_movie.original_title.values[0])
    def getNeighbors(baseMovie, K):
        distances = []

        for index, movie in movies.iterrows():
            if movie['new_id'] != baseMovie['new_id'].values[0]:
                dist = Similarity(baseMovie['new_id'].values[0], movie['new_id'])
                distances.append((movie['new_id'], dist))

        distances.sort(key=operator.itemgetter(1))
        neighbors = []

        for x in range(K):
            neighbors.append(distances[x])
        return neighbors

    K = 10
    avgRating = 0
    neighbors = getNeighbors(new_movie, K)

    print('\nRecommended Movies: \n')
    for neighbor in neighbors:
        avgRating = avgRating+movies.iloc[neighbor[0]][2]
        print( movies.iloc[neighbor[0]][0]+" | Genres: "+str(movies.iloc[neighbor[0]][1]).strip('[]').replace(' ','')+" | Rating: "+str(movies.iloc[neighbor[0]][2]))

    print('\n')
    avgRating = avgRating/K
    print('The predicted rating for %s is: %f' %(new_movie['original_title'].values[0],avgRating))
    print('The actual rating for %s is %f' %(new_movie['original_title'].values[0],new_movie['vote_average']))




if __name__ == '__main__':
    # print('calling preview')
    # preview()
    #insert Screen 1
    # print('calling jsontostring')
    jsontostring()
    #insert Screen 2
    movies=movies.merge(mov,left_on='id',right_on='movie_id',how='left')# merging the two csv files
    movies=movies[['id','original_title','genres','cast','vote_average','director','keywords']]
    movies['genres']=movies['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
    movies['genres']=movies['genres'].str.split(',')
    plt.subplots(figsize=(12,10))
    list1=[]
    for i in movies['genres']:
        list1.extend(i)
    ax=pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9)
    for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values):
        ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
    ax.patches[9].set_facecolor('r')
    plt.title('Top Genres')
    # plt.show()
    #Insert Graph
    for i,j in zip(movies['genres'],movies.index):
        list2=[]
        # list2=i
        list2.sort()
        movies.loc[j,'genres']=str(list2)
    movies['genres']=movies['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
    movies['genres']=movies['genres'].str.split(',')
    genreList = []
    for index, row in movies.iterrows():
        genres = row["genres"]

        for genre in genres:
            if genre not in genreList:
                genreList.append(genre)
    # print(genreList[:10]) #now we have a list with unique genres
    #Binary columns of genres
    movies['genres_bin'] = movies['genres'].apply(lambda x: binary(x))
    # print(movies['genres_bin'].head(4))
    #Insert Screen 4
    #Working with the Cast Column
    #50k unique values, as many movies have entries for about 15-20 actors. But do we need all of them??
    #Luckily,the the sequence of the actors in the JSON format is according to the actor's importance.
    movies['cast']=movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
    movies['cast']=movies['cast'].str.split(',')
    plt.subplots(figsize=(12,10))
    list1=[]
    for i in movies['cast']:
        list1.extend(i)
    ax=pd.Series(list1).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9)
    for i, v in enumerate(pd.Series(list1).value_counts()[:15].sort_values(ascending=True).values):
        ax.text(.8, i, v,fontsize=10,color='white',weight='bold')
    plt.title('Actors with highest appearance')
    ax.patches[14].set_facecolor('r')
    # plt.show()
    #Insert screen 5
    for i,j in zip(movies['cast'],movies.index):
        list2=[]
        list2=i[:4]
        movies.loc[j,'cast']=str(list2)
    movies['cast']=movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')
    movies['cast']=movies['cast'].str.split(',')
    for i,j in zip(movies['cast'],movies.index):
        list2=[]
        list2=i
        list2.sort()
        movies.loc[j,'cast']=str(list2)
    movies['cast']=movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')
    movies['cast']=movies['cast'].str.split(',')
    castList = []
    for index, row in movies.iterrows():
        cast = row["cast"]

        for i in cast:
            if i not in castList:
                castList.append(i)
    movies['cast_bin'] = movies['cast'].apply(lambda x: binary(x))
    # movies['cast_bin'].head(2)
    #Insert screen 6
    #Working with the director column
    movies['director']=movies['director'].apply(xstr)
    plt.subplots(figsize=(12,10))
    ax=movies[movies['director']!=''].director.value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.85)
    for i, v in enumerate(movies[movies['director']!=''].director.value_counts()[:10].sort_values(ascending=True).values):
        ax.text(.5, i, v,fontsize=12,color='white',weight='bold')
    ax.patches[9].set_facecolor('r')
    plt.title('Directors with highest movies')
    # plt.show()
    #Insert Screen 7
    directorList=[]
    for i in movies['director']:
        if i not in directorList:
            directorList.append(i)
    movies['director_bin'] = movies['director'].apply(lambda x: binary(x))
    # print(movies['director_bin'].head(2))
    #Insert screen 8
    #Working with the keywords column
    plt.subplots(figsize=(12,12))
    stop_words=set(stopwords.words('english'))
    stop_words.update(',',';','!','?','.','(',')','$','#','+',':','...',' ','')
    words=movies['keywords'].dropna().apply(nltk.word_tokenize)
    movies['keywords']=movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
    movies['keywords']=movies['keywords'].str.split(',')
    for i,j in zip(movies['keywords'],movies.index):
        list2=[]
        list2=i
        movies.loc[j,'keywords']=str(list2)
    movies['keywords']=movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'')
    movies['keywords']=movies['keywords'].str.split(',')
    for i,j in zip(movies['keywords'],movies.index):
        list2=[]
        list2=i
        list2.sort()
        movies.loc[j,'keywords']=str(list2)
    movies['keywords']=movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'')
    movies['keywords']=movies['keywords'].str.split(',')
    words_list = []
    for index, row in movies.iterrows():
        keyws = row["keywords"]

        for keyw in keyws:
            if keyw not in words_list:
                words_list.append(keyw)
    movies['words_bin'] = movies['keywords'].apply(lambda x: binary(x))
    # print(movies['words_bin'].head(2))
    #Insert Screen 9
    movies=movies[(movies['vote_average']!=0)] #removing the movies with 0 score and without drector names
    movies=movies[movies['director']!='']
    # Checking similarity between movies
    # Defined a function
    #Checking similarity between 2 random movies
    # print(Similarity(3,160))
    # print(Similarity(152,2))
    # Insert Screen 11
    #Checking what the movies actually were
    # print(movies.iloc[3])
    # print(movies.iloc[160])
    # Insert Screen 10
    #Turned out to be The Dark Knight Rises and How to train your Dragon which are very different
    new_id=list(range(0,movies.shape[0]))
    movies['new_id']=new_id
    movies=movies[['original_title','genres','vote_average','genres_bin','cast_bin','new_id','director','director_bin','words_bin']]
    # print(movies.head(2))
    # Insert Screen 12
    
    # So now when we have everything in place, we will now build the score predictor. The main function working under the hood will be the Similarity function, which will calculate the similarity between movies, and will find 10 most similar movies. These 10 movies will help in predicting the score for our desired movie. We will take the average of the scores of the similar movies and find the score for the desired movie.
    # Now the similarity between the movies will depend on our newly created columns containing binary lists. We know that features like the director or the cast will play a very important role in the movie's success. We always assume that movies from David Fincher or Chris Nolan will fare very well. Also if they work with their favorite actors, who always fetch them success and also work on their favorite genres, then the chances of success are even higher. Using these phenomena, lets try building our score predictor.
    print(whats_my_score('Godfather'))
    print(whats_my_score('Minions'))
    print(whats_my_score('Dark Knight'))
    print(whats_my_score('Balboa'))
    #Insert Screen 13
