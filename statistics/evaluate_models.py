"""
Evaluate three recommenders using Recall at K = 9
"""

# Import packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import dok_matrix, csr_matrix
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
import pandas as pd
import numpy as np


# Function declarations
def find_most_similar_posts_collabfilter(comments, k=9):
    user_post = comments[['from_id', 'post_id']]
    user_post.drop_duplicates(inplace=True)
    unique_users = user_post.from_id.drop_duplicates().values
    unique_posts = user_post.post_id.drop_duplicates().values
    users_map = dict(zip(unique_users, range(len(unique_users))))
    posts_map = dict(zip(unique_posts, range(len(unique_posts))))
    user_all_posts = user_post.groupby('from_id')['post_id'].apply(list).reset_index()

    item_matrix = dok_matrix((len(unique_users), len(unique_posts)), dtype=np.float32)
    for l in range(user_all_posts.shape[0]):
        i = users_map[user_all_posts.iloc[l].from_id]
        posts = user_all_posts.iloc[l].post_id
        for post in posts:
            j = posts_map[post]
            item_matrix[i, j] = 1

    cosine_sim = cosine_similarity(item_matrix.transpose())

    similar_posts = []
    similar_rating = []
    for l in range(cosine_sim.shape[0]):
        source_sim = cosine_sim[l,:]
        sim_ids = np.argpartition(source_sim, -k)[-k:]
        sim = source_sim[sim_ids]
        sim_posts = [unique_posts[d] for d in sim_ids]
        similar_posts.append(sim_posts)
        similar_rating.append(sim)

    df = pd.DataFrame(data = {'post_id': unique_posts,
                             'most_similar': similar_posts,
                             'most_similar_rating': similar_rating})

    return (df)


def split_id(val):
    return(re.sub("[^_,\d]", "", val).split(","))


def split_rating(val, sep=","):
    if sep == " ":
        patterns = "[\[\]]"
        string_rating = re.sub(patterns, "", val).split()
    else:
        patterns = "[^.,\d]"
        string_rating = re.sub(patterns, "", val).split(",")
    float_rating = [float(r) for r in string_rating]
    return(float_rating)


def format_similarity(df, sep=","): #split string to post_id's and similarities
    df['most_similar'] = df.most_similar.apply(split_id)
    df['most_similar_rating'] = df.most_similar_rating.apply(split_rating, sep=sep)
    return(df)


def is_in_top_k(test_case, rec, train_set, k=9):
    user = test_case[0]
    truth = test_case[1]
    user_old_posts = train_set.loc[user]
    recs = rec[rec.post_id.isin(user_old_posts)].most_similar.values.tolist()
    recs = np.array([np.array(sublist) for sublist in recs]).flatten()
    recs_rating = rec[rec.post_id.isin(user_old_posts)].most_similar_rating.values.tolist()
    recs_rating = np.array([np.array(sublist) for sublist in recs_rating]).flatten()
    top_k_indx = np.argpartition(recs_rating, -k)[-k:]
    if truth in recs[top_k_indx]:
        return 1
    else:
        return 0


def evaluate_models(comments, n_fold=5):
    doc2vec = format_similarity(pd.read_csv("data/fb_news_posts_20K_doc2v.csv"))
    tfidf = format_similarity(pd.read_csv("data/fb_news_posts_20K_tfidf.csv"), sep=" ")

    test_prob = 1/n_fold
    if n_fold == 1:
        test_prob = 0.2

    # 5 fold cross validation
    comments = comments.drop_duplicates(['from_id', 'post_id'])
    users = comments.groupby('from_id').count().iloc[:, 0].reset_index() # created_time : number of posts read
    potential_test_users = users[users.created_time > 1] # only include users who have read > 1 posts in test set
    test_set_size = np.int(test_prob*potential_test_users.shape[0])

    doc2vec_cv_recall = []
    tfidf_cv_recall = []
    cf_cv_recall = []
    cv_recalls = [doc2vec_cv_recall, tfidf_cv_recall, cf_cv_recall]

    for n in range(n_fold):

        test_users = potential_test_users.iloc[(n*test_set_size): ((n+1)*test_set_size - 1), :]
        test_set = pd.merge(test_users, comments.drop_duplicates('from_id'), on='from_id').loc[:,['from_id', 'post_id']]

        train_set = pd.merge(test_set, comments, on=['from_id','post_id'], how='outer', indicator=True)
        train_set = train_set[train_set['_merge'] != 'both'].loc[:,['from_id',  'post_id']]
        train_set_unique = train_set.groupby('from_id')['post_id'].apply(list)

        cf = find_most_similar_posts_collabfilter(train_set)

        recommenders = [doc2vec, tfidf, cf]

        # recommendation results
        for m in range(len(recommenders)):
            r = test_set.apply(is_in_top_k,
                              rec=recommenders[m],
                              train_set=train_set_unique,
                              axis=1)
            cv_recalls[m].append(sum(r)/len(r))

    return( cv_recalls)


# Calculate cross-validated recall rates of three models
posts = pd.read_csv("../data/fb_news_posts_20K.csv")
comments = pd.read_csv("../data/fb_news_comments_1000k_cleaned.csv")
cv_recalls = evaluate_models(comments)
# np.save("cv_recalls", cv_recalls) # save output
