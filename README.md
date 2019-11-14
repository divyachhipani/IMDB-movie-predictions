# ObjectDetection
Machine Learning project to predict movie's IMDB rating
IMDB Movie Reviews Sentimental Analysis

Problem Statement-
Predict IMDb movie ratings based on attributes obtained from the dataset. It is a simple classification problem. 
We need to use Machine Learning to train a model that takes the attribute of the movie as input and predicts it’s IMDB ratings. 
We used a linear classifier: Logistic Regression and coded the model using Python

Dataset-
The dataset is of around 46 MB. It has around 5000 rows of data. Kaggle has removed the original version of this dataset per a DMCA takedown request from IMDB. 
They offer a replaced it with a similar set of films and data fields from The Movie Database (TMDb) Link:
https://www.kaggle.com/tmdb/tmdb-movie-metadata
Files:
../input/tmdb_5000_credits.csv:
Contains information of the cast and crew for each movie.
Columns: Movie_id- Numeric, Title- String, Cast- String, Crew- String ../input/tmdb_5000_movies.csv
Contains information like the score, title, date_of_release, genres, etc.
Columns: budget- Numeric, genres- String, homepage- String, id- Numeric, keywords- String, original_language- String, original_title- String, overview- String, popularity- Numeric, production_companies- String, production_countries- String, release_date- DateTime, revenue- Numeric, runtime- Numeric, spoken_languages- String, status- String, tagline- String, title- String, vote_average- Numeric, vote_count- Numeric
The main problem with this dataset is the .json format. Many columns in the dataset are in json format, therefore cleaning the dataset was the main challenge. For people who don't know about JSON(JavaScript Object Notation), it is basically a syntax for storing and exchanging data between two computers. It is mainly in a key:value format, and is embedded into a string.

Code-
We know that the score of a movie depends on various factors like the genre, or the actor working in the film and mainly the director of the film. 
Considering all such factors, we will try to build a simple score predictor for this dataset.

 
Results-
The model worked out pretty well as the predicted scores are very close to the actual scores. Although our approach was very simple, the model is working fine. 
The score prediction could be more accurate if we had more data.

