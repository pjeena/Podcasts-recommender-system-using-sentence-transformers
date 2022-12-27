import numpy as np
import pandas as pd
import pickle
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
import sys
sys.path.insert(1, '/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push')
from config import PODCASTS_DATABASE_PATH_RAW, USER_REVIEWS_DATABASE_PATH_RAW, PODCASTS_DATABASE_PATH_PROCESSED, USER_REVIEWS_DATABASE_PATH_PROCESSED

from preprocessing import get_preprocessed_text
from utils import get_recommendation_tfidf
from utils import get_cosine_similarity_podcasts, get_manhattan_distance_products, get_euclidean_distances_products

# Loading data 
df_podcasts = pd.read_csv('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/data/processed/podcasts.csv')

tfidf_embeddings_matrix = scipy.sparse.load_npz('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/models/tfidf_embeddings_matrix.npz')
#tfidf_embeddings_matrix = pickle.load(open('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/models/tfidf_embeddings_matrix.pkl', 'rb'))
similarity_matrix = pickle.load(open('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/models/similarity_matrix.pkl', 'rb'))

podcast_id = input('Enter the podcast name :: ')
#podcast_id = "Something Was Wrong"
recommendations = get_recommendation_tfidf(podcast_id, df_podcasts,'cosine',tfidf_embeddings_matrix,n=10)

names = [i['value'] for i in recommendations]

podcasts_recommendations = df_podcasts.set_index('name').loc[names].reset_index()[['name','category']]

import plotly.graph_objects as go
import pandas as pd
fig = go.Figure(data=[go.Table(
    header=dict(values=list(podcasts_recommendations.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[podcasts_recommendations.name, podcasts_recommendations.category],
               fill_color='lavender',
               align='left'))
])
fig.show()
