from .models import Post, Comment, UserTest
from django.contrib.auth.models import User
from sklearn.metrics.pairwise import *
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np

def update_filter():
    # Create a sparse matrix of user - post
    # element[i][j] filled with 1 when a user i comment on post j
    all_user_names = list(map(lambda x: x.name, UserTest.objects.all()))
    all_post_ids = list(map(lambda x: x.id, Post.objects.all()))
    num_users = len(all_user_names)
    num_posts = len(all_post_ids)

    comment_matrix = dok_matrix((num_users, num_posts), dtype=np.float32)
    for i in range(num_users):
        user_comments = Comment.objects.filter(from_name=all_user_names[i])
        for user_comment in user_comments:
            j = all_post_ids.index(user_comment.post_id.id)
            comment_matrix[i, j] = 1

    # Calculate pairwise similarity between posts
    cosine_sim = cosine_similarity(comment_matrix.transpose())

    # Update cosine CosineSimilarity
    CosineSimilarity.objects.all().delete()
    for i in range(num_posts):
        source_id = all_post_ids[i]
        for j in range(i+1, num_posts):
            target_id = all_post_ids[j]
            new_similarity = CosineSimilarity(source_id=source_id, target_id=target_id, similarity=cosine_sim[i,j])
            new_similarity.save()
