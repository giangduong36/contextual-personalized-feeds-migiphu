from django.contrib.auth.models import User
from sklearn.metrics.pairwise import *
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'recsys')))
from recsys.models import Post, Comment, UserTest, CosineSimilarity

def update_filter():
    # Create a sparse matrix of user - post
    # element[i][j] filled with 1 when a user i comment on post j
    all_user_names = list(map(lambda x: x.name, UserTest.objects.all()))
    all_post_ids = list(map(lambda x: x.id, Post.objects.all()))
    num_users = len(all_user_names)
    num_posts = len(all_post_ids)
    print('Number of posts is \n', num_posts, file=sys.stderr)
    print('Number of users is \n', num_users, file=sys.stderr)

    post_id_index_map = dict(zip(all_post_ids, np.arange(num_posts)))
    user_name_index_map = dict(zip(all_user_names, np.arange(num_users)))
    print('Loading user - list of posts into memory \n', file=sys.stderr)
    all_users_posts = Comment.objects.raw('SELECT id, from_name, GROUP_CONCAT(post_id_id) as post_ids \
    FROM recsys_comment \
    GROUP BY from_name')
    print('Done \n', file=sys.stderr)

    comment_matrix = dok_matrix((num_users, num_posts), dtype=np.float32)
    for user_posts in all_users_posts:
        i = user_name_index_map[user_posts.from_name]
        posts = user_posts.post_ids.split(",")
        for post in posts:
            j = post_id_index_map[post]
            comment_matrix[i, j] = 1
    print('Made a user-post matrix!!!!!!\n', file=sys.stderr)

    # Calculate pairwise similarity between posts
    cosine_sim = cosine_similarity(comment_matrix.transpose())
    print('Calculated cosine similarity!!!!!!!!! \n', file=sys.stderr)

    # Update cosine CosineSimilarity
    CosineSimilarity.objects.all().delete()
    n = 30 # set this manually to reduce computation load
    for i in range(num_posts):
        source_id = all_post_ids[i]
        source_sim = cosine_sim[i,:]
        top_n = np.argpartition(source_sim, -n)[-n:]
        for j in top_n:
            target_id = all_post_ids[j]
            new_similarity = CosineSimilarity(source_id=source_id, target_id=target_id, similarity=cosine_sim[i,j])
            new_similarity.save()
    print('Update cosine similarities table!!!!!!!!! \n', file=sys.stderr)
    return(1)
