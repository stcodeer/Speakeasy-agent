import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import Levenshtein
import re
from collections import Counter


ml_ratings = pd.read_csv('data/ratings.csv', usecols=['userId', 'movieId', 'rating'])
ml_matrix = ml_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
ml_csr = csr_matrix(ml_matrix.values)

movie_df = pd.read_csv('data/movies.csv', usecols=['movieId', 'title', 'genres'])

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=17)
knn.fit(ml_csr[2:, :])

def func_3(input_title):
    genres = []
    flag = 0
    genres_2 = []

    for x in input_title:
        min_distance = float('inf')
        most_similar_title = None
        most_similar_index = None
        for index, row in movie_df.iterrows():
            title = row['title']
            distance = Levenshtein.distance(x, title[:-6])
        
            if distance < min_distance:
                min_distance = distance
                most_similar_title = title
                most_similar_index = index
        genres.append(movie_df.loc[most_similar_index, 'genres'])

        if most_similar_title:
            pattern = re.compile(rf"^{re.escape(x)}\b", re.IGNORECASE)
            for _, row in movie_df.iterrows():
                if pattern.match(row['title']) and row['title'] != most_similar_title:
                    flag = 1
                    break

    year = most_similar_title[-5:-1]

    all_genres = []
    for genre in genres:
        all_genres.extend(genre.split('|'))

    genre_counter = Counter(all_genres)
    ans_genre = genre_counter.most_common(1)[0][0]

    for x in input_title:
        min_distance_2 = float('inf')
        most_similar_title_2 = None
        most_similar_index_2 = None
        for index, row in movie_df.iterrows():
            title = row['title']
            distance = Levenshtein.distance(x, title[:-6])
        
            if distance < min_distance_2:
                min_distance_2 = distance
                most_similar_title_2 = title
                most_similar_index_2 = index

        distance, indices = knn.kneighbors(ml_csr[most_similar_index_2], n_neighbors=1)
        for i in indices[0]:
            # print(movie_df['title'].iloc[i])
            most_similar_title_2 = movie_df['title'].iloc[i]
            genres_2.append(movie_df['genres'][i])        

    year_2 = most_similar_title_2[-5:-1]

    all_genres_2 = []
    for genre in genres_2:
        all_genres_2.extend(genre.split('|'))

    genre_counter_2 = Counter(all_genres)
    ans_genre_2 = genre_counter_2.most_common(1)[0][0]

    if flag:
        answer_template = "Adequate recommendations will be {} movies from the {}s or sequel to the movies.".format(ans_genre,year)
    else:
        answer_template = "Adequate recommendations will be {} movies from the {}s.".format(ans_genre,year)

    return answer_template


if __name__ == '__main__':
    input_title = ["Hamlet","Othello"]
    print(func_3(input_title))
