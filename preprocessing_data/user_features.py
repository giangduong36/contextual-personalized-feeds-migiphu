import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import math

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def filter_less_active():
    user_df = pd.read_csv('../data//fb_news_users_features.csv')
    small = user_df.loc[(user_df['size'] > 5) & (~ (user_df['gender']).isnull())]
    # quantiles = np.linspace(0, 1, 10).tolist()
    # quantile_edges = np.unique(small['size'].quantile(quantiles)).tolist()

    small['active_level'] = pd.cut(small['size'], [0, 6, 7, 8, 9, 12, 16, 22, 417], labels=np.arange(1, 9))

    small.to_csv(path_or_buf='../data//fb_news_users_features_active.csv',
                   index=False)


def prepare_user_features():
    # Get gender based on first name (given existing genders file)
    gender_df = pd.read_csv('../data//genders.csv')
    user_df = pd.read_csv('../data//fb_news_users.csv', dtype=str)
    genderList = []
    allnames = list(user_df['name'])
    cnt = 0
    for name in tqdm(allnames):
        if cnt % 100 == 0: print(cnt)
        cnt += 1
        firstname = str(name).split(" ")[0]
        gender = gender_df.loc[gender_df['name'] == firstname]['gender'].values
        if len(gender) == 0:
            genderList.append('')
        else:
            genderList.append(gender[0])

    user_df['gender'] = genderList
    user_df.info()

    num_comments = []

    comment_dict = load_obj('comment_dict')
    for i, r in user_df.iterrows():
        num_comments.append(comment_dict[r.id])




    # # Number of comments
    # comment_df = pd.read_csv('data//fb_news_comments_1000k_cleaned.csv', dtype=str)
    # size_comment = comment_df.groupby('from_id').size().reset_index(name='size')
    # user_df = user_df.merge(size_comment, left_on='id', right_on='from_id', how='inner')
    # user_df.drop(columns=['from_id'], inplace=True)

    # Assign an active level from 1-10 (based on how many comments)
    user_df['active_level'] = pd.cut(user_df['size'], 10, labels=np.arange(1, 11))

    # print(user_df.sort_values('size', ascending=False).head(10))
    print(user_df.groupby('active_level').size())

    user_df.to_csv(path_or_buf='data//fb_news_users_features.csv',
                   index=False)


def extract_features(active_level=10):
    user_features = {}

    # gender
    genders = ['male', 'female']

    # active_level (based on number of comments)
    # active_level = np.arange(1, 11).astype(str).tolist()
    active_level = [('active' + str(j)) for j in np.arange(1, active_level+1).astype(str).tolist()]

    sample_feature_vec = dict.fromkeys(genders + active_level, 0)
    print(sample_feature_vec)
    user_data = pd.read_csv('../data//fb_news_users_features_active.csv', dtype=str)

    for i, row in user_data.iterrows():
        if i % 100 == 0:
            print(i)
        user_id = row['id']
        gender = row['gender']
        active = row['active_level']

        user_features[user_id] = sample_feature_vec.copy()
        if gender in user_features[user_id]:  # Avoid nan
            user_features[user_id][gender] = 1
        user_features[user_id]['active' + str(active)] = 1

    with open('../data//user_features_engineered_small.csv', 'w') as w:
        w.write('user_id' + ',' + ','.join(str(x) for x in genders + active_level) + '\n')
        for user_id, features in user_features.items():
            w.write(user_id + ',' + ','.join(str(x) for x in features.values()) + '\n')


if __name__ == "__main__":
    filter_less_active()
    # prepare_user_features()
    extract_features(active_level=8)
