import pandas as pd
import numpy as np


if __name__ == "__main__":
    comment_data = pd.read_csv('../data/fb_news_comments_1000k_cleaned.csv', dtype=str)

    # print(comment_data.columns.tolist())
    user_post_commented = comment_data[['from_id', 'post_id']]
    user_post_commented.columns = ['user_id', 'post_id']
    print(user_post_commented.info())

    user_post_commented.drop_duplicates(subset=['user_id', 'post_id'],inplace=True)

    user_post_commented.to_csv(path_or_buf='../data/user_post_commented.csv',
                               index=False)
    print(user_post_commented.info())
