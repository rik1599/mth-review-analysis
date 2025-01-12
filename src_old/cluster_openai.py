"""
    Description: This file contains:
        - get_embedding: function that returns the embedding of a text using the OpenAI API
        - clustering_openai: function that applies the get_embedding function to each row of the dataframe
        - normalize_l2: function that normalizes the embedding of a text
        - shape_data: function that prints the shape of the dataframe and the shape of the ada_embedding field
        - main: function that applies the clustering_openai function to each topic and feature

    Author : Gallegos Carvajal Ian Marco
"""

from config import API_KEY
from openai import OpenAI
import pandas as pd
import numpy as np

from utils import set_feature_np
from plot import elbow_method, silhouette_method, visualize_clusters, bar_plot

from sentence_transformers import SentenceTransformer

client = OpenAI(api_key=API_KEY)


def get_embedding(text):
    #model = 'text-embedding-3-small'
    text = text.replace("\n", " ")
    print(text)
    response = client.embeddings.create(
        model="text-embedding-3-large", input=text, encoding_format="float"
    )

    cut_dim = response.data[0].embedding[:256]
    return cut_dim


def clustering_openai(df, feature, df_out):
    df_test = df.copy()
    # create new columns in the dataframe
    df_test['ada_embedding'] = "None"
    df_test['ada_embedding_normalized'] = "None"

    #apply get_embedding function to each row of the dataframe
    for i in range(len(df_test)):
        df_test['ada_embedding'].iloc[i] = get_embedding(df_test[feature].iloc[i])
        df_test['ada_embedding_normalized'].iloc[i] = normalize_l2(df_test['ada_embedding'].iloc[i])

    df_test.to_csv(df_out, index=False)


def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


def shape_data(df):
    print(f"Shape df: {df.shape}")
    # For each column in the dataframe, print the shape of the column
    for col in df.columns:
        print(f"Shape {col}: ")#{len(df[col].iloc[0])}")

    # for each row in the dataframe, print the len of the ada_embedding field
    for i in range(len(df)):
        print(f"Emb: {len(df['ada_embedding'].iloc[i])} - Norm: {len(df['ada_embedding_normalized'].iloc[i])}")


def analyze_clusters():
    topic = 'covid' # 'climate' #
    feature = 'post_description' # 'general_topic' #

    image_out = topic + '_' + feature
    df = pd.read_csv('results/' + image_out + '_' + 'cluster.csv')
    # set print options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    # remove ada_embedding and ada_embedding_normalized columns
    df = df.drop(columns=['ada_embedding', 'ada_embedding_normalized'])
    # print the first 10 rows of the dataframe row by row
    for i in range(10):
        print(f"search_query: {df.iloc[i]['search_query']}")
        print(f"post_description: {df.iloc[i]['post_description']}")
        print(f"general_topic: {df.iloc[i]['general_topic']}")
        print('='*100)


def main():
    # topic = 'covid' # 'climate' #
    # feature = 'search_query' # 'post_description' # 'general_topic' #
    #
    # df = pd.read_csv('MUSMA-Project/' + topic + '_gpt_results.csv')
    # df_out = 'csv/' + topic + '_' + feature + '_' + 'LLM_gpt_Emb_openai' + '.csv'
    # # clustering_openai(df, feature, df_out)
    #
    # feature_emb = 'ada_embedding_normalized' # 'ada_embedding'
    # image_out = topic + '_' + feature
    # df = pd.read_csv(df_out)
    #
    # # convert to numpy array the feature
    # set_feature_np(df, feature_emb)
    #
    # n_cluster = elbow_method(df, feature_emb, image_out)
    # silhouette_method(df, feature_emb, image_out + '_' + 'silhouette_method.png')
    #
    # visualize_clusters(df, feature, feature_emb, n_cluster, image_out)

    # shape_data(df)
    for topic in ['covid', 'climate']:
        for feature in ['search_query', 'post_description', 'general_topic']:
            image_out = topic + '_' + feature
            df = pd.read_csv('results/' + image_out + '_' + 'cluster.csv')
            bar_plot(df, "Cluster", image_out)


def sentence_similarity():
    sentence_source = "That is a happy person"
    sentences_compare = ["That is a happy dog", "That is a very happy person", "Today is a sunny day"]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentence_source_embedding = model.encode(sentence_source)
    sentences_compare_embedding = model.encode(sentences_compare)

    for i, sentence in enumerate(sentences_compare):
        # round to 3 decimal places the similarity
        similarity = round(np.dot(sentence_source_embedding, sentences_compare_embedding[i]), 3)
        print(f"Similarity: " + str(similarity) + " - Sentence: {sentence}")



if __name__ == '__main__':
    main()