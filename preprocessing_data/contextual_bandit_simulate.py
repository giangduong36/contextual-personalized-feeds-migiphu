import datetime
import math
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

"""
An implementation of contextual bandit LinUCB algorithm from paper: Li et al. (2010)
Can run on both sample contextual bandit dataset or a sampling dataset from Facebook News Dataset
"""


# Helper function to save Python object for easy and quick retrieval later
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def create_dataset(num_user=1000, num_post=10, num_cat=5):
    user_data = pd.read_csv('../data/user_features_engineered_small.csv', dtype=str)
    post_data = pd.read_csv('../data//post_features_engineered_small_set_cat.csv')

    users_sample = user_data[user_data.user_id.isin(pd.Series(user_data.user_id.unique()).sample(num_user))].copy()

    users_sample.reset_index(drop=True, inplace=True)
    # comment_sample = comment_data[comment_data.user_id.isin(users_sample.user_id)].copy()
    # articles_sample = post_data[post_data.post_id.isin(comment_sample.post_id)].copy()
    # print(len(articles_sample))

    # Load an existing dictionary with user jds as keys, posts a user commented on as values
    user_comment_dict = load_obj('comment_dict')

    dataset = pd.DataFrame()

    n_comments = []

    for user_id in tqdm(users_sample.user_id, mininterval=10):
        articles_sample_commented = post_data[post_data.post_id.isin(user_comment_dict[user_id])].copy()
        n_comments.append(len(articles_sample_commented))

        # 1st way to sample non-commented posts for each user in sample set
        post_data_limit_cat = post_data[post_data.category.isin(pd.Series(post_data.category.unique()).sample(num_cat))]
        articles_sample_random = post_data_limit_cat[
            post_data_limit_cat.post_id.isin(pd.Series(post_data_limit_cat.post_id.unique()).sample(num_post)) &
            (~post_data_limit_cat.post_id.isin(user_comment_dict[user_id]))].copy()
        articles_sample = articles_sample_commented.append(articles_sample_random)

        # # 2nd way to sample non-commented posts for each user in sample set
        # # (better randomization but more time consuming)
        # # Number of commented posts = 1/10 # uncommented with similar size for each category
        # articles_sample = articles_sample_commented.copy()
        # commented_cat_dist = articles_sample_commented.groupby('category').size()
        # # print(commented_cat_dist).tolist()
        # for cat in commented_cat_dist.index:
        #     articles_sample_random = post_data.loc[post_data.category == cat]
        #     articles_sample_random = articles_sample_random[
        #         articles_sample_random.post_id.isin(
        #             pd.Series(articles_sample_random.post_id.unique()).sample(num_post)) &
        #         (~articles_sample_random.post_id.isin(user_comment_dict[user_id]))].copy()
        #     articles_sample = articles_sample.append(articles_sample_random)
        # # print(articles_sample.shape)

        articles_sample.reset_index(drop=True, inplace=True)

        index = users_sample.index[users_sample.user_id == user_id].tolist()[0]
        repeat_user_row = users_sample.loc[np.repeat(index, len(articles_sample))]
        repeat_user_row.reset_index(drop=True, inplace=True)
        merge = pd.concat([articles_sample, repeat_user_row], axis=1, ignore_index=True)
        # Add a reward column: 1 if the user commented on the post, 0 otherwise
        reward = np.concatenate(
            [np.repeat(1, len(articles_sample_commented)),
             np.repeat(0, len(articles_sample) - len(articles_sample_commented))])
        merge['reward'] = reward
        dataset = dataset.append(merge)

    # rearrange columns
    newcolnames = post_data.columns.tolist()
    newcolnames.extend(user_data.columns.tolist())
    newcolnames.append('reward')
    dataset.columns = newcolnames
    cols = dataset.columns.tolist()
    cols = [cols[0]] + [cols[-1]] + cols[1:-1]
    dataset = dataset[cols]
    dataset.drop(columns='user_id', inplace=True)

    print("Dataset reward distribution: ", dataset.groupby('reward').size())

    print("Dataset category distribution: ", dataset.groupby('category').size())

    print("Number of comments", n_comments)

    dataset.to_csv(
        path_or_buf='../data/dataset_bandit_active' + str(num_user) + '_' + str(num_post) + 'small_set_cat_v2.csv',
        index=False)
    return dataset


def reduce_dimension_pca(posts_raw, users_raw, wanted_dim=6):
    pca_df = posts_raw[list(posts_raw.columns[1:])]
    pca = PCA(n_components=wanted_dim)
    pca_df = pd.DataFrame(pca.fit_transform(pca_df))
    pca_df.index = posts_raw.index
    pca_df['category'] = posts_raw.category

    # Percentage of variance explained by each of the selected components.
    pca.explained_variance_ratio_.sum()

    posts = pd.concat([posts_raw[list(posts_raw.columns[:1])], pd.DataFrame(pca_df)], axis=1)

    pca_df_user = users_raw[list(users_raw.columns[1:])]
    pca = PCA(n_components=wanted_dim)
    pca_df_user = pd.DataFrame(pca.fit_transform(pca_df_user))
    pca_df_user.index = users_raw.index

    users = pd.concat([users_raw[list(users_raw.columns[:1])], pd.DataFrame(pca_df_user)], axis=1)

    # print(posts_raw.shape, (users_raw.shape), posts.shape, users.shape)
    cols = pca_df.columns.tolist()
    cols = ['category', 'reward'] + cols[0:-2]
    pca_df = pca_df[cols]

    return posts, users


# Choose the next action (article to serve a user) for bandit
def choose_action(A, b, context, alpha, POST):
    best_value = -math.inf
    best_a = None

    for a in POST:
        A_a_inv = np.linalg.inv(A[a])
        theta_a = A_a_inv.dot(b[a])
        p = theta_a.transpose().dot(context)
        p += alpha * math.sqrt(context.transpose().dot(A_a_inv).dot(context))

        if p > best_value:
            best_value = p
            best_a = a

    return best_a


# Run contextual bandit LinUCB on a sample dataset from Facebook News Dataset
def run_generated_dataset(dataset):
    POST = (dataset['category']).unique()
    # Since the Facebook News Dataset is very sparse, suggest a whole category of posts instead of individual posts

    actions = np.array(dataset['category'])
    # dataset.drop(columns=['post_id'], inplace=True)
    # dataset.drop(columns=['category'], inplace=True)

    rewards = np.array(dataset['reward'].astype(int).tolist())
    contexts = np.array(dataset.iloc[:, 2:].astype(float).as_matrix())

    print(len(actions))
    print(len(rewards))
    print(contexts.shape)
    # T = trial t
    T = len(dataset)
    D = contexts.shape[1]

    shuff = np.arange(T)
    np.random.shuffle(shuff)
    actions = actions[shuff]
    rewards = rewards[shuff]
    contexts = contexts[shuff]

    run_bandit(POST, D, T, actions, rewards, contexts,
               resultname="CRT on generated dataset " + str(datetime.datetime.now()) + ".png")


# Function to test LinUCB on the sample dataset from Professor Jebara, Columbia University
def run_sample_dataset():
    dataset = open("../data/bandit_test_dataset.txt")  # Source: Professor Jebara, Columbia University
    actions = np.zeros(10000)
    rewards = np.zeros(10000)
    contexts = np.zeros((10000, 100))

    T = 0  # Number of trials
    for line in dataset:
        line = line.split(" ")[:-1]  # dataset.txt

        actions[T] = float(line[0])
        rewards[T] = float(line[1])
        # rewards[T] = 1 #classification.txt
        for i in range(100):
            contexts[T][i] = float(line[i + 2])
        T += 1

    POST = np.arange(1, 11)
    D = 100
    run_bandit(POST, D, T, actions, rewards, contexts,
               resultname="CTR on sample dataset " + str(datetime.datetime.now()) + ".png"
               )


def run_bandit(POST, D, T, actions, rewards, contexts, resultname='CTR Result'):
    """
    Run LinUCB algorithm and save the cumulative take rate to an image file
    :param POST:
    :param D:
    :param T:
    :param actions:
    :param rewards:
    :param contexts:
    :param resultname:
    :return:
    """
    # initialize variables
    theta, A, b = {}, {}, {}

    for i in POST:
        A[i] = np.identity(D)
        b[i] = np.zeros((D, 1))
        theta[i] = np.random.normal(size=D)

    num, deno, payoff, valid_total = 0, 0, 0, 0
    CTR, best_a_list = [], []

    alpha = 1

    # LinUCB algorithm
    for t in tqdm(range(1, T + 1)):
        alpha = 1 / np.sqrt(t)
        p = {}

        # calculate CRT
        arm_chosen = choose_action(A, b, contexts[t - 1], alpha, POST)
        num += (int(rewards[t - 1])) * (actions[t - 1] == arm_chosen)
        deno += (actions[t - 1] == arm_chosen)

        if deno == 0:
            CTR.append(0)
        else:
            CTR.append(num / deno)

        # find arm with highest upper confidence bound
        A_t = np.array(POST)

        for a in A_t:
            A_a_inv = np.linalg.inv(A[a])
            theta[a] = A_a_inv.dot(b[a])
            p[a] = theta[a].transpose().dot(contexts[t - 1])
            p[a] += alpha * math.sqrt(contexts[t - 1].transpose().dot(A_a_inv).dot(contexts[t - 1]))

        best_value = -math.inf
        best_a = None
        for a in A_t:
            if p[a] > best_value:
                best_value = p[a]
                best_a = a

        # print(len(A_t), best_a, best_value, actions[t-1])
        best_a_list.append(best_a)
        # only query reward if we choose the correct arm

        if best_a == actions[t - 1]:
            print("Valid: ", best_a, actions[t - 1], rewards[t - 1])
            reward = rewards[t - 1]
            temp = contexts[t - 1].reshape((D, 1))  # TODO
            A[best_a] += temp.dot(temp.transpose())
            b[best_a] += reward * temp

            payoff += reward
            valid_total += 1

    # print and plot results
    print(" " + str(num / deno))

    print("Valid predictions in total ", valid_total)
    if valid_total != 0:
        print(payoff / valid_total)
    print("-----")

    print("Frequency best articles: ", Counter(best_a_list))

    # Plot and save the graph of cumulative take rate to file
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('alpha = ' + str(alpha) + ", final CTR = " + str(num / deno))
    ax.set_xlabel('T')
    ax.set_ylabel('Cumulative take rate')
    plt.plot(CTR)

    plt.savefig(resultname)
    plt.gcf().clear()


if __name__ == "__main__":
    comment_dict = load_obj('comment_dict')

    run_sample_dataset()

    # # Create a new sample of users and contextual information
    # n_user = 5000
    # n_article = 20
    # dataset = create_dataset(n_user, n_article)

    # Use an existing dataset with 1000 users and
    dataset = pd.read_csv('../data/dataset_bandit_active1000_10_set_cat_v1.csv')

    run_generated_dataset(dataset)
