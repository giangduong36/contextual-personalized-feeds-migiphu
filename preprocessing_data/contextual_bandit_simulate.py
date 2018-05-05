import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.decomposition import PCA


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def choose_action(A, b, context, alpha):
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


def create_dataset(num_user=1000, num_post=10, num_cat=5):
    user_data = pd.read_csv('../data/user_features_engineered_small.csv', dtype=str)
    post_data = pd.read_csv('../data//post_features_engineered_small_set_cat.csv')

    # post_data, user_data = reduce_dimension_pca(post_data_orig, user_data_orig)
    # comment_data = pd.read_csv('../data//user_post_commented.csv', dtype=str)

    users_sample = user_data[user_data.user_id.isin(pd.Series(user_data.user_id.unique()).sample(num_user))].copy()
    # print("Users number of comments: ", users_sample.groupby('size').size())

    # users_sample = user_data.copy()
    users_sample.reset_index(drop=True, inplace=True)
    # comment_sample = comment_data[comment_data.user_id.isin(users_sample.user_id)].copy()
    # articles_sample = post_data[post_data.post_id.isin(comment_sample.post_id)].copy()
    # print(len(articles_sample))

    user_comment_dict = load_obj('comment_dict')

    dataset = pd.DataFrame()

    n_comments=[]

    for user_id in tqdm(users_sample.user_id, mininterval=10):
        articles_sample_commented = post_data[post_data.post_id.isin(user_comment_dict[user_id])].copy()
        n_comments.append(len(articles_sample_commented))

        # 1st way to sample non-commented posts
        # post_data_limit_cat = post_data[post_data.category.isin(pd.Series(post_data.category.unique()).sample(num_cat))]
        # articles_sample_random = post_data_limit_cat[
        #     post_data_limit_cat.post_id.isin(pd.Series(post_data_limit_cat.post_id.unique()).sample(num_post)) &
        #     (~post_data_limit_cat.post_id.isin(user_comment_dict[user_id]))].copy()
        # articles_sample = articles_sample_commented.append(articles_sample_random)

        # 2nd attempt to make # commented = 1/10 # uncommented with similar size for each category
        articles_sample = articles_sample_commented.copy()
        commented_cat_dist = articles_sample_commented.groupby('category').size()
        # print(commented_cat_dist).tolist()
        for cat in commented_cat_dist.index:
            articles_sample_random = post_data.loc[post_data.category == cat]
            articles_sample_random = articles_sample_random[
                articles_sample_random.post_id.isin(
                    pd.Series(articles_sample_random.post_id.unique()).sample(num_post)) &
                (~articles_sample_random.post_id.isin(user_comment_dict[user_id]))].copy()
            articles_sample = articles_sample.append(articles_sample_random)
        # print(articles_sample.shape)

        articles_sample.reset_index(drop=True, inplace=True)

        index = users_sample.index[users_sample.user_id == user_id].tolist()[0]
        repeat_user_row = users_sample.loc[np.repeat(index, len(articles_sample))]
        # repeat_user_row.drop(columns='user_id', inplace=True)
        repeat_user_row.reset_index(drop=True, inplace=True)
        merge = pd.concat([articles_sample, repeat_user_row], axis=1, ignore_index=True)
        reward = np.concatenate(
            [np.repeat(1, len(articles_sample_commented)), np.repeat(0, len(articles_sample) - len(articles_sample_commented))])
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
    # user_id = dataset['user_id']
    dataset.drop(columns='user_id', inplace=True)
    # dataset.insert(0, 'user_id', user_id)

    print("Dataset reward distribution: ", dataset.groupby('reward').size())

    print("Dataset category distribution: ", dataset.groupby('category').size())

    print("Number of comments", n_comments)

    dataset.to_csv(path_or_buf='../data/dataset_bandit_active' + str(num_user) + '_' + str(num_post) + 'small_set_cat_v2.csv',
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

    # print(posts_raw.shape)
    # print(users_raw.shape)
    # print(posts.shape)
    # print(users.shape)
    cols = pca_df.columns.tolist()
    cols = ['category', 'reward'] + cols[0:-2]
    pca_df = pca_df[cols]

    return posts, users


def reduce_dim():
    dataset = pd.read_csv('../data/dataset_bandit.csv')
    featuredf = dataset.iloc[:, 2:]
    pca = PCA(n_components=6)
    pca_df = pd.DataFrame(pca.fit_transform(featuredf))
    pca_df.index = dataset.index
    pca_df['category'] = dataset.category
    pca_df['reward'] = dataset.reward
    cols = pca_df.columns.tolist()
    cols = ['category', 'reward'] + cols[0:-2]
    pca_df = pca_df[cols]
    pca_df.to_csv(path_or_buf='../data/dataset_bandit_pca.csv',
                   index=False)


def split_data_equal(dataset):
    pass


if __name__ == "__main__":
#     n_user = 1000
#     n_article = 20
#
#     dataset = create_dataset(n_user, n_article)
#

    comment_dict = load_obj('comment_dict')
# # def main2():
    n_user = 5000
    n_article = 20
# #
#     dataset = create_dataset(n_user, n_article)
# # #
# #     # dataset = pd.read_csv('../data/dataset_bandit_active' + str(n_user) + '_' + str(n_article) + '.csv')
# #     dataset= pd.read_csv('../data/dataset_bandit_active5000_40small_set_cat_v1.csv')
# #
#     # dataset.sort_values(by=['hour', 'day'], inplace=True)
#
#     print("Dataset reward distribution: ", dataset.groupby('reward').size())
#
#     print("Dataset category distribution: ", dataset.groupby('category').size())  # dataset = dataset.drop_duplicates()
#
#     print(dataset.shape)
#
#     # POST = dataset['post_id'].unique()
#     POST = (dataset['category']).unique()
#     N_POST = len(POST)
#     # actions = np.array(dataset['category'], dtype=str)
#     actions = np.array(dataset['category'])
#     # dataset.drop(columns=['post_id'], inplace=True)
#     # dataset.drop(columns=['category'], inplace=True)
#
#     # actions = np.array(dataset.index//N_POST)
#
#     rewards = np.array(dataset['reward'].astype(int).tolist())
#     contexts = np.array(dataset.iloc[:, 2:].astype(float).as_matrix())
#
#     print(len(actions))
#     print(len(rewards))
#     print(contexts.shape)
#     # print(actions)
#     # print(contexts)
#     # T = trial t
#     T = len(dataset)
#     D = contexts.shape[1]
#
#     shuff = np.arange(T)
#     np.random.shuffle(shuff)
#     actions = actions[shuff]
#     rewards = rewards[shuff]
#     contexts = contexts[shuff]

    # # EXISTING DATASET
    #
    dataset = open("dataset.txt")
    # dataset = open("classification.txt")
    actions = np.zeros(10000)
    rewards = np.zeros(10000)
    contexts = np.zeros((10000, 100))

    T = 0
    for line in dataset:
        line = line.split(" ")[:-1]   #dataset.txt
        # line = line.split(" ")    #classification.txt
        # line = line.split(" ")
        # line[len(line)-1] = line[len(line)-1][:-1]

        actions[T] = float(line[0])
        rewards[T] = float(line[1])
        # rewards[T] = 1 #classification.txt
        for i in range(100):
            # print(line[i + 1])
            contexts[T][i] = float(line[i + 2])
        T += 1

    POST = np.arange(1,11)
    D = 100

    # initialize variables
    theta = {}
    A = {}
    b = {}

    for i in POST:
        A[i] = np.identity(D)
        b[i] = np.zeros((D, 1))
        theta[i] = np.random.normal(size=D)

    num = 0
    deno = 0
    CTR = []

    payoff = 0
    valid_total = 0

    best_a_list = []

    # LinUCB algorithm
    for t in tqdm(range(1, T + 1)):
        alpha = 1 / np.sqrt(t)

        p = {}

        # calculate CRT
        arm_chosen = choose_action(A, b, contexts[t - 1], alpha)
        # print(rewards[t - 1], type(rewards[t - 1]))
        num += (int(rewards[t - 1])) * (actions[t - 1] == arm_chosen)
        deno += (actions[t - 1] == arm_chosen)

        a = 0
        if deno == 0:
            CTR.append(0)
        else:
            a = num / deno
            CTR.append(num / deno)

        # print(a)
        # find arm with highest upper confidence bound
        # A_t = [i for i in range(1, N_POST)]
        # A_t = [actions[t - 1]]
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

            # print(best_a, best_value, valid_total)

    # print and plot results
    print(" " + str(num / deno))

    print("valid total ", valid_total)
    if valid_total != 0:
        print(payoff / valid_total)
    print("-----")

    print("Frequency best articles: ", Counter(best_a_list))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('alpha = ' + str(alpha) + ", final CTR = " + str(num / deno))
    ax.set_xlabel('T')
    ax.set_ylabel('Cumulative take rate')
    plt.plot(CTR)

    import datetime
    plt.savefig(str(datetime.datetime.now()) + "_" + str(n_user) + "_" + str(n_article) + "_"+ str(alpha) + "v2.png")
    plt.gcf().clear()



    # Number of users have commented more than 5 times: 28200
