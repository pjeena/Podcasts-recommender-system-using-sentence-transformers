import streamlit as st
import scipy.sparse
import json
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import sentence_transformers
from utils import get_recommendation_tfidf
from preprocessing import get_preprocessed_text
from utils import get_cosine_similarity_podcasts, get_manhattan_distance_products, get_euclidean_distances_products

@st.cache(suppress_st_warning=True, hash_funcs={dict: lambda _: None, pd.DataFrame: lambda _: None}, allow_output_mutation=True)
def data_loading():
    df_podcasts = pd.read_csv('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/data/processed/podcasts.csv')
    df_podcasts_top_10 = pd.read_csv('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/data/processed/Top_10_podcasts.csv')
    tfidf_embeddings_matrix = scipy.sparse.load_npz('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/models/tfidf_embeddings_matrix.npz')
    similarity_matrix = pickle.load(open('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/models/similarity_matrix.pkl', 'rb'))
    values_podcast_names = df_podcasts.name.values.tolist()
    values_podcast_names.insert(0,'Type your choice.....')
    model_miniLM = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    minilm_embeddings_matrix = pickle.load(open('/Users/piyush/Desktop/dsml_Portfolio/podcast/final_push/models/minilm_embeddings_matrix.pkl', 'rb'))

    return df_podcasts, df_podcasts_top_10,model_miniLM, tfidf_embeddings_matrix,similarity_matrix,values_podcast_names,minilm_embeddings_matrix

df_podcasts, df_podcasts_top_10, model_miniLM, tfidf_embeddings_matrix,similarity_matrix,values_podcast_names,minilm_embeddings_matrix  = data_loading()

siteHeader = st.container()
dataExploration = st.container()

with siteHeader:
    st.title('Podcast Recommendation System')
    st.markdown("by :red[**Piyush Jeena**] :computer:")

with dataExploration:
    user_choice = st.selectbox('**How would you like to get podcast recommendations?**',
       options = ('Select an option.....','Select a podcast I have already heard', 'Enter podcast description'))

    if user_choice == 'Select a podcast I have already heard':
        podcast_id = st.selectbox('**Select podcast**',options = values_podcast_names,index=0)

        if podcast_id != 'Type your choice.....':
            n = st.slider('**How many recommendations would you like?**', 
                                min_value=1, max_value=20, 
                            value=5, 
                                step=1)
            recommendations = get_recommendation_tfidf(podcast_id, df_podcasts,'cosine',tfidf_embeddings_matrix,n)

            names = [i['value'] for i in recommendations]

            podcasts_recommendations = df_podcasts.set_index('name').loc[names].reset_index()[['name','category']]
            podcasts_recommendations.rename(columns={"name": "Podcast", "category": "Category"},inplace=True)
            styler = podcasts_recommendations.style.hide(axis='index')
            st.write(styler.to_html(), unsafe_allow_html=True)  


    elif user_choice == 'Enter podcast description':
        podcast_keywords_from_user = st.text_input('**Enter keywords**')

        if podcast_keywords_from_user:
            input_embedding = model_miniLM.encode(podcast_keywords_from_user).reshape(1,-1)
            scores = cosine_similarity(minilm_embeddings_matrix,input_embedding)
            n = st.slider('**How many recommendations would you like?**', min_value=1, max_value=20, value=5,step=1)

            sorted_scores = sorted(list(enumerate(scores)),key=lambda x:x[1],reverse=True)[0:n]
            recommendations =  [{'value': df_podcasts.iloc[x[0]]['name'], 'score' : np.round(x[1], 2)} for x in sorted_scores]
            names = [i['value'] for i in recommendations]
            podcasts_recommendations = df_podcasts.set_index('name').loc[names].reset_index()[['name','category']]
            podcasts_recommendations.rename(columns={"name": "Podcast", "category": "Category"},inplace=True)
            styler = podcasts_recommendations.style.hide(axis='index')
            st.write(styler.to_html(), unsafe_allow_html=True)

with st.sidebar.expander("", expanded=True):
        st.markdown(':blue[**Top 10 Podcasts based on Ratings**]')
        fig = go.Figure(data=[go.Table(
               columnwidth=[0.3,0.5],
                header=dict(values=list(df_podcasts_top_10.columns),
                fill_color='paleturquoise',
                align='left'),
                cells=dict(values=[df_podcasts_top_10['Podcast'],df_podcasts_top_10['Ratings']],
               fill_color='lavender',
               align='left'))
            ])
        fig.update_layout(margin=dict(l=2,r=2,b=2,t=5))
        st.write(fig)
