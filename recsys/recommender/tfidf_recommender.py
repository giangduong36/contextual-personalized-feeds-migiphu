# Based off of:
# https://stackoverflow.com/questions/37593293/what-is-the-simplest-way-to-get-tfidf-with-pandas-dataframe
# https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity

# from recsys.models import *
# # from django.contrib.auth.models import User
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

# Find three most similar posts to each post that a user has commented on
# def tfidf_recommend_posts(user_id):
#     user = UserTest.objects.get(id=user_id)
#     comments = Comment.objects.filter(from_id=user_id)
#     commented_post = [c.post_id for c in comments]
"""
Run tfidf on fb_news_posts_20k.csv and return 3 most similar posts for each post. 
Save as a csv for later use
Will be fixed later for official integration with Users
"""


def find_most_similar_posts_tfidf(k=3):
    df = pd.read_csv('../../data/fb_news_posts_20K.csv')[['post_id', 'message']]
    df.fillna('', inplace=True)
    tfidf = TfidfVectorizer().fit_transform(df['message'])
    similar = []
    for i in range(0, df.shape[0]):
        cosine_similarities = linear_kernel(tfidf[i:i + 1], tfidf).flatten()
        # sim_index = cosine_similarities.argsort()[-2]   # Get the most similar post
        sim_ids = cosine_similarities.argsort()[-2:-(2 + k):-1]  # The the 3 most similar posts
        rec_posts = []
        for sim_index in sim_ids:
            postid = df.ix[sim_index]['post_id']
            rec_posts.append(postid)
        similar.append(rec_posts)
    df['most_similar'] = similar
    df.to_csv(path_or_buf='../../data/fb_news_posts_20K_tfidf.csv',
              index=False,
              columns=['post_id', 'most_similar'])

    return df


# def find_most_similar_tfidf(df):
#     similar = []
#     for i in range(0, df.shape[0]):
#         cosine_similarities = linear_kernel(tfidf[i:i+1], tfidf).flatten()
#         sim_index = cosine_similarities.argsort()[-2]   # Get the most similar post
#         sim_ids = cosine_similarities.argsort()[-2:-5:-1]  # The the 3 most similar posts
#         print(cosine_similarities.argsort(), sim_ids)
#         text = df.ix[sim_index]['message']
#         similar.append(text)
#     df['most_similar'] = similar
#     return df
#

# df = pd.read_csv('../../data/fb_news_posts_20K.csv')[['post_id', 'message']]
# df.fillna('', inplace=True)
#
# tfidf = TfidfVectorizer().fit_transform(df['message'])

# print(find_most_similar_posts_tfidf().head())
