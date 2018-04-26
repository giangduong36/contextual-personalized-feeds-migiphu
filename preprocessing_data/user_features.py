import pandas as pd
import numpy as np


def prepare_user_features():
    # Get gender based on first name (given existing genders file)
    gender_df = pd.read_csv('data//genders.csv')
    user_df = pd.read_csv('data//fb_news_users.csv', dtype=str)
    genderList = []
    allnames = list(user_df['name'])
    cnt = 0
    for name in allnames:
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

    # Number of comments
    comment_df = pd.read_csv('data//fb_news_comments_1000k_cleaned.csv', dtype=str)
    size_comment = comment_df.groupby('from_id').size().reset_index(name='size')
    user_df = user_df.merge(size_comment, left_on='id', right_on='from_id', how='inner')
    user_df.drop(columns=['from_id'], inplace=True)

    # Assign an active level from 1-10 (based on how many comments)
    user_df['active_level'] = pd.cut(user_df['size'], 10, labels=np.arange(1, 11))

    # print(user_df.sort_values('size', ascending=False).head(10))
    print(user_df.groupby('active_level').size())

    user_df.to_csv(path_or_buf='data//fb_news_users_features.csv',
                   index=False)


def extract_features():
    user_features = {}

    # gender
    genders = ['male', 'female']

    # degree of engagement
    degree_online_engagement = np.arange(1, 11).astype(str).tolist()
    print(type(degree_online_engagement[0]))

    sample_feature_vec = dict.fromkeys(genders + degree_online_engagement, 0)
    print(sample_feature_vec)
    user_data = pd.read_csv('data//fb_news_users_features.csv', dtype=str)

    for i, row in user_data.iterrows():
        if i % 100 == 0:
            print(i)
        user_id = row['id']
        gender = row['gender']
        active = row['active_level']

        user_features[user_id] = sample_feature_vec.copy()
        if gender in user_features[user_id]:  # Avoid nan
            user_features[user_id][gender] = 1
        user_features[user_id][active] = 1

    with open('data//user_features_engineered.csv', 'w') as w:
        for user_id, features in user_features.items():
            w.write(user_id + ',' + ','.join(str(x) for x in features.values()) + '\n')


if __name__ == "__main__":
    prepare_user_features()
    extract_features()
