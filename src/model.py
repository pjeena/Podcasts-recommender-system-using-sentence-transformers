import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sentence_transformers
import scipy.sparse
import sys
sys.path.insert(1, '/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push')
from config import PODCASTS_DATABASE_PATH_RAW, USER_REVIEWS_DATABASE_PATH_RAW, PODCASTS_DATABASE_PATH_PROCESSED, USER_REVIEWS_DATABASE_PATH_PROCESSED
from preprocessing import get_preprocessed_text
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from tqdm import tqdm


# Loading data 
df_podcasts = pd.read_csv(PODCASTS_DATABASE_PATH_RAW,header=None)
df_users = pd.read_csv(USER_REVIEWS_DATABASE_PATH_RAW,header=None)

columns_pod = ['id','name','url','studio','category','episode_count','avg_rating','total_ratings','description']
columns_users = ['id','podcasts_id','username','review_title','review','rating','date']

df_podcasts.columns = columns_pod
df_users.columns = columns_users
df_users = df_users.drop('id', axis=1)

df_podcasts['tags'] =  df_podcasts['description'] + ' ' + df_podcasts['category']
df_podcasts['tags_parsed'] =  df_podcasts['tags'].apply(lambda x: get_preprocessed_text(x)) 
tags_parsed_list = list(df_podcasts['tags_parsed'])

model_tfidf = TfidfVectorizer(max_features=1000,stop_words='english', analyzer='word', ngram_range=(1,3))
tfidf_embeddings_matrix = model_tfidf.fit_transform(tags_parsed_list)
similarity_matrix = cosine_similarity(tfidf_embeddings_matrix)
#pickle.dump(tfidf_embeddings_matrix, open('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/models/tfidf_embeddings_matrix.pkl', 'wb'))
scipy.sparse.save_npz('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/models/tfidf_embeddings_matrix.npz', tfidf_embeddings_matrix)
pickle.dump(similarity_matrix, open('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/models/similarity_matrix.pkl', 'wb'))


model_miniLM = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
minilm_embeddings_matrix = []
for text in tqdm(df_podcasts['description']):
        minilm_embeddings_matrix.append(model_miniLM.encode(text))
minilm_embeddings_matrix = np.array(minilm_embeddings_matrix)
pickle.dump(minilm_embeddings_matrix, open('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/models/minilm_embeddings_matrix.pkl', 'wb'))