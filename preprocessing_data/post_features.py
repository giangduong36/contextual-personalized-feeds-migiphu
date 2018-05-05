import pandas as pd
import numpy as np
import dateutil.parser
from sklearn.decomposition import PCA

"""
Python script to extract features of posts for contextual bandit algorithm from Facebook News Dataset
Features included: Length of post's message, Number of each type of reactions, Number of shares, Post's created time
"""

def prepare_post_features():
    post_df = pd.read_csv('../data/fb_news_posts_20K.csv')
    # post_df = post_df.applymap(str)
    post_df['message'] = post_df['message'].astype(str)

    # Length of posts
    messages = post_df['message'].tolist()
    num_words_arr = [len(i.split(" ")) for i in messages]
    # post_df['cat_length'] = pd.cut(num_words_arr, 10, labels=np.arange(1, 11))

    post_df['cat_length'] = num_words_arr

    # Number of each reaction, shares

    reactions = ['react_angry', 'react_haha', 'react_like', 'react_love', 'react_sad', 'react_wow', 'shares']
    # get categories for each reaction
    cut_to_categories(post_df, reactions, bins=5)

    # Create time

    time_features = ['hour', 'day', 'month']

    get_time_features(post_df, time_features)
    # get exact value each reaction
    post_df.to_csv(path_or_buf='../data/fb_news_posts_features_.csv',
                   index=False)


def cut_to_categories(df, feature_list, bins=5):
    quantiles = np.linspace(0, 1, bins).tolist()
    for feature in feature_list:
        quantile_edges = np.unique(df[feature].quantile(quantiles)).tolist()
        labels = np.arange(1, len(quantile_edges)).astype(int).astype(str).tolist()
        df['cat_' + feature] = pd.cut(df[feature], quantile_edges, labels=labels, include_lowest=True)


def get_time_features(df, feature_list):
    for feature in feature_list:
        print("Loading features....: " + feature)
        df[feature] = [getattr(dateutil.parser.parse(i), feature) for i in df['created_time']]


# Build a data frame of features
def extract_features():
    post_df = pd.read_csv('../data/fb_news_posts_features.csv')

    post_features = {}

    # number of reactions
    reactions = ['react_angry', 'react_haha', 'react_like', 'react_love', 'react_sad', 'react_wow', 'shares']
    react_cat = ['cat_' + i for i in reactions]

    react_dict = {}
    for i in range(len(react_cat)):
        react_dict[react_cat[i]] = reactions[i]

    # created_time
    time_features = ['hour', 'day', 'month']

    # all_features = ['cat_length'] + reactions + time_features
    all_features = ['cat_length'] + react_cat + time_features

    all_features_val = []
    #
    for feature in all_features:
        all_features_val.extend([(feature + str(j)) for j in np.unique(post_df[feature]).tolist()])

    # all_features_val=all_features
    # print(all_features_val)

    sample_feature_vec = dict.fromkeys(all_features_val, 0)
    print(sample_feature_vec)

    # A csv file of features with exact data
    post_data = pd.read_csv('../data/fb_news_posts_features.csv')
    post_data['message'] = post_data['message'].astype(str)

    for i, row in post_data.iterrows():
        if i % 100 == 0:
            print(i)
        post_id = row['post_id']

        post_features[post_id] = sample_feature_vec.copy()
        for feature in all_features:
            if feature in react_dict:
                post_features[post_id][feature + str(row[feature])] = row[react_dict[feature]]
            elif feature in time_features:
                post_features[post_id][feature + str(row[feature])] = row[feature]
            elif feature == 'cat_length':
                post_features[post_id][feature + str(row[feature])] = len(row['message'].split(" "))
            else:
                post_features[post_id][feature + str(row[feature])] = 1
            # post_features[post_id][feature] = row[feature]

    with open('../data/post_features_engineered_cat_exact.csv', 'w') as w:
        w.write('post_id' + ',' + ','.join(str(x) for x in all_features_val) + '\n')
        for post_id, features in post_features.items():
            w.write(post_id + ',' + ','.join(str(x) for x in features.values()) + '\n')


if __name__ == "__main__":
    # prepare_post_features()
    extract_features()
    # post_data = pd.read_csv('data/fb_news_posts_features.csv', dtype=str)

