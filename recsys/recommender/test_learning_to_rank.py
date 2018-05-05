import re
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import dok_matrix, csr_matrix
from tqdm import tqdm
import collections
import random
from collections import Counter

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def split_id(val):
    return (re.sub("[^_,\d]", "", val).split(","))


def split_rating(val, sep=","):
    if sep == " ":
        patterns = "[\[\]]"
        string_rating = re.sub(patterns, "", val).split()
    else:
        patterns = "[^.,\d]"
        string_rating = re.sub(patterns, "", val).split(",")
    float_rating = [float(r) for r in string_rating]
    return (float_rating)


def find_most_similar_posts_collabfilter(comments, k=9):
    user_post = comments[['from_id', 'post_id']]
    user_post.drop_duplicates(inplace=True)
    unique_users = user_post.from_id.drop_duplicates().values
    unique_posts = user_post.post_id.drop_duplicates().values
    users_map = dict(zip(unique_users, range(len(unique_users))))
    posts_map = dict(zip(unique_posts, range(len(unique_posts))))
    user_all_posts = user_post.groupby('from_id')['post_id'].apply(list).reset_index()

    item_matrix = dok_matrix((len(unique_users), len(unique_posts)), dtype=np.float32)
    for l in tqdm(range(user_all_posts.shape[0])):
        i = users_map[user_all_posts.iloc[l].from_id]
        posts = user_all_posts.iloc[l].post_id
        for post in posts:
            j = posts_map[post]
            item_matrix[i, j] = 1

    cosine_sim = cosine_similarity(item_matrix.transpose())

    similar_posts = []
    similar_rating = []
    for l in tqdm(range(cosine_sim.shape[0])):
        source_sim = cosine_sim[l, :]
        sim_ids = np.argpartition(source_sim, -k)[-k:]
        sim = source_sim[sim_ids]
        sim_posts = [unique_posts[d] for d in sim_ids]
        similar_posts.append(sim_posts)
        similar_rating.append(sim)

    df = pd.DataFrame(data={'post_id': unique_posts,
                            'most_similar': similar_posts,
                            'most_similar_rating': similar_rating})
    return df


def format_similarity(df, sep=","):  # split string to post_id's and similarities
    df['most_similar'] = df.most_similar.apply(split_id)
    df['most_similar_rating'] = df.most_similar_rating.apply(split_rating, sep=sep)
    return (df)


doc2vec = format_similarity(pd.read_csv("../../data/fb_news_posts_20K_doc2v.csv"))
tfidf = format_similarity(pd.read_csv("../../data/fb_news_posts_20K_tfidf.csv"), sep=" ")
comments = pd.read_csv("../../data/fb_news_comments_1000k_cleaned.csv")
# cf = find_most_similar_posts_collabfilter(comments)
cf = load_obj('collab_filtering')

features = {'1': doc2vec, '2': tfidf, '3':cf}
comment_dict = load_obj('comment_dict')
users = [i for i in comment_dict.keys() if len(comment_dict[i]) > 5]    # Select active users
users=random.sample(users, 1)

# final = pd.DataFrame(columns=['commented', 'user_id', 'f1:s1', 'f2:s2', 'f3:s3', 'post_id'])

# tfidf_dict = load_obj('tfidf_dict')
# cf_dict = load_obj('cf_dict')
# doc2vec_dict = load_obj('doc2vec_dict')
# features = {'1': doc2vec_dict, '2': tfidf_dict, '3':cf_dict}

final = collections.defaultdict(lambda: collections.defaultdict(int))

for user in tqdm(users):
    for commented_post in comment_dict[user]:
        # Add commented_post
        final[(user, commented_post)]['commented'] = 2
        # Add similar post
        for f_index in features:
            f = features[f_index]
            f_similar = f.loc[f.post_id == commented_post]['most_similar'].values[0]
            f_rating = f.loc[f.post_id == commented_post]['most_similar_rating'].values[0]

            for similar, rating in zip(f_similar, f_rating):
                final[(user, similar)][f_index] = rating
                final[(user, similar)]['commented'] = int((similar) in comment_dict[user]) + 1


# import dill
# dill.dumps(final, 'final' + str(len(final)))

commented = [i['commented'] for i in final.values()]

qid = [i[0] for i in final]

f1 = [i['1'] for i in final.values()]
f2 = [i['2'] for i in final.values()]
f3 = [i['3'] for i in final.values()]
eid = [i[1] for i in final]

final_df = pd.DataFrame()
final_df['commented'] = commented
final_df['qid'] = qid
final_df['f1'] = f1
final_df['f2'] = f2
final_df['f3'] = f3
final_df['eid'] = eid

df = final_df[['f1', 'f2', 'f3']]
normalized_df=(df-df.min())/(df.max()-df.min())
final_df['f1'] = normalized_df.f1
final_df['f2'] = normalized_df.f2
final_df['f3'] = normalized_df.f3

save_obj(final_df, 'final_df' + str(len(users)))

train, validate, test = np.split(final_df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])


def write_to_file(name, df):
    with open(name, 'w') as file:
        for i, r in tqdm(df.iterrows()):
            file.write(str(r['commented']) + ' ')
            file.write('qid:' + str(r['qid']) + ' ')
            file.write('1:' + str(r['f1']) + ' ')
            file.write('2:' + str(r['f2']) + ' ')
            file.write('3:' + str(r['f3']) + ' ')
            file.write('#eid = ' + str(r['eid']) + '\n')

write_to_file('train' + str(len(users)) +'.txt', train)
write_to_file('validate' + str(len(users)) +'.txt', validate)
write_to_file('test' + str(len(users)) +'.txt', test)


def test(test_path, score_path):
    test = pd.read_csv(test_path, header=None, sep=' ')
    score = pd.read_csv(score_path, header=None, sep='\t')
    score['eid'] = test[7]
    score.columns = ['uid', '1', 'score', 'eid']
    best10 = score.sort_values(['uid','score'],ascending=False).groupby('uid').head(10)
    test_eid = best10.groupby(by='uid').eid.apply(list)
    test_eid.index = test_eid.index.astype(str)
    test_score = best10.groupby(by='uid').score.apply(list)
    test_score.index = test_score.index.astype(str)


    counts = []
    ndcg_scores = []
    for i in tqdm(test_eid.index):
        cnt=0
        ytrue = []
        yscore = test_score[i]
        for post in test_eid[i]:
            if post in comment_dict[i]:
                ytrue.append(1)
                cnt +=1
            else:
                ytrue.append(0)
        ndcg_scores.append(ndcg_score(ytrue, yscore))
        counts.append(cnt)

    print("Average recall: ", np.mean(counts))
    print(Counter(counts))
    print("ndcg_Score", np.nanmean(ndcg_scores), ndcg_scores)


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains * 1.0 / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual * 1.0 / best


# test('test1000.txt', 'score.txt')
test('test5000.txt', 'score5000.txt')

