# Based off of:
# https://stackoverflow.com/questions/37593293/what-is-the-simplest-way-to-get-tfidf-with-pandas-dataframe
# https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity

# from recsys.models import *
# # from django.contrib.auth.models import User
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

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
    vec = TfidfVectorizer(stop_words='english')
    tfidf = vec.fit_transform(df['message'])
    similar = []
    similar_scores = []
    for i in tqdm(range(0, df.shape[0])):
        cosine_similarities = linear_kernel(tfidf[i:i + 1], tfidf).flatten()
        sim_ids = cosine_similarities.argsort()[-2:-(2 + k):-1]  # The the k most similar posts
        sim = cosine_similarities[sim_ids]
        rec_posts = []
        for sim_index in sim_ids:
            postid = df.iloc[sim_index]['post_id']
            rec_posts.append(postid)
        similar.append(rec_posts)
        similar_scores.append(sim)
    df['most_similar'] = similar
    df['most_similar_rating'] = similar_scores
    df.to_csv(path_or_buf='../../data/fb_news_posts_20K_tfidf.csv',
              index=False,
              columns=['post_id', 'most_similar', 'most_similar_rating'])

    ## Split posts into clusters based on tf-idf
    # number_of_clusters = 15
    # km = KMeans(n_clusters=number_of_clusters)
    # km.fit(tfidf)
    # print("Top terms per cluster:")
    # order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    # terms = vec.get_feature_names()
    # for i in range(number_of_clusters):
    #     top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
    #     print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))


find_most_similar_posts_tfidf(k=9)

