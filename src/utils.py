import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('wordnet',quiet=True)
nltk.download('omw-1.4',quiet=True)

def get_cosine_similarity_podcasts(df,similarity_matrix, index, n):
    
    # calculate cosine similarity between each vectors
    result = list(enumerate(similarity_matrix[index]))
    
    # Sorting the Score
    sorted_result = sorted(result,key=lambda x:x[1],reverse=True)[1:n+1]
    
    similar_podcasts =  [{'value': df.iloc[x[0]]['name'], 'score' : round(x[1], 2)} for x in sorted_result]
     
    return similar_podcasts




def get_euclidean_distances_products(df,similarity_matrix, index, n): 

    # Getting Score and Index
    result = list(enumerate(similarity_matrix[index]))

    # Sorting the Score and taking top 10 products
    sorted_result = sorted(result,key=lambda x:x[1],reverse=False)[1:n+1]

    # Mapping index with data
    similar_podcasts =  [{'value': df.iloc[x[0]]['name'], 'score' : round(x[1], 2)} for x in sorted_result]
    
    return similar_podcasts



def get_manhattan_distance_products(df,similarity_matrix, index, n):   
     
    # Getting Score and Index
    result = list(enumerate(similarity_matrix[index]))

    # Sorting the Score and taking top 10 products
    sorted_result = sorted(result,key=lambda x:x[1],reverse=False)[1:n+1]
    
    # Mapping index with data
    similar_podcasts =  [{'value': df.iloc[x[0]]['name'], 'score' : round(x[1], 2)} for x in sorted_result]
    
    return similar_podcasts



# Comparing similarity to get the top matches using TF-IDF

def get_recommendation_tfidf(podcast_id, df, similarity, tfidf_embeddings_matrix, n):

    row = df.loc[df['name'] == podcast_id]
    index = list(row.index)[0]
 #   description = row['tags_parsed'].loc[index]
 #   tags_list = list(df['tags_parsed'])
    #Create vector using tfidf
    
 #   tfidf_matrix = tfidf_vec.fit_transform(tags_list)
    
    if similarity == "cosine":
        sim_matrix = cosine_similarity(tfidf_embeddings_matrix)
        podcasts = get_cosine_similarity_podcasts(df,sim_matrix , index,n)
        
    elif similarity == "manhattan":
        sim_matrix = manhattan_distances(tfidf_embeddings_matrix)
        podcasts = get_manhattan_distance_products(df,sim_matrix , index,n)
        
    else:
        sim_matrix = euclidean_distances(tfidf_embeddings_matrix)
        podcasts = get_euclidean_distances_products(df,sim_matrix , index,n)

    return podcasts