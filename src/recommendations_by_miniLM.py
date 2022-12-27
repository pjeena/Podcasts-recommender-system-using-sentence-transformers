import numpy as np
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
import sys
sys.path.insert(1, '/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push')
from config import PODCASTS_DATABASE_PATH_RAW, USER_REVIEWS_DATABASE_PATH_RAW, PODCASTS_DATABASE_PATH_PROCESSED, USER_REVIEWS_DATABASE_PATH_PROCESSED
import sentence_transformers
from preprocessing import get_preprocessed_text
from utils import get_recommendation_tfidf
from utils import get_cosine_similarity_podcasts, get_manhattan_distance_products, get_euclidean_distances_products

# Loading data 
df_podcasts = pd.read_csv('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/data/processed/podcasts.csv')
minilm_embeddings_matrix = pickle.load(open('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/models/minilm_embeddings_matrix.pkl', 'rb'))




podcast_keywords_from_user = input('Enter your keywords :: ')
podcast_keywords_from_user = get_preprocessed_text(podcast_keywords_from_user)

model_miniLM = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
input_embedding = model_miniLM.encode(podcast_keywords_from_user).reshape(1,-1)

scores = cosine_similarity(minilm_embeddings_matrix,input_embedding)
n = 10
sorted_scores = sorted(list(enumerate(scores)),key=lambda x:x[1],reverse=True)[0:n]
recommendations =  [{'value': df_podcasts.iloc[x[0]]['name'], 'score' : np.round(x[1], 2)} for x in sorted_scores]
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