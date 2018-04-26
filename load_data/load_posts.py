import sys, os
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "recsys440.settings")

import django

django.setup()

from recsys.models import Post, Page


def save_post_from_row(post_row):
    post = Post()
    post.created_time = post_row[0]
    post.description = post_row[1]
    post.link = post_row[2]
    post.message = post_row[3]
    post.page_id = Page.objects.get(id=post_row[4])
    post.id = post_row[5]
    post.react_angry = post_row[6]
    post.react_haha = post_row[7]
    post.react_like = post_row[8]
    post.react_love = post_row[9]
    post.react_sad = post_row[10]
    post.react_wow = post_row[11]
    post.scrape_time = post_row[12]
    post.shares = post_row[13]
    post.save()


def delete_db():
    print('truncate db')
    Post.objects.all().delete()
    print('finished truncate db')


if __name__ == "__main__":

    if len(sys.argv) == 2:

        delete_db()
        print("Reading from file " + str(sys.argv[1]))
        posts_df = pd.read_csv(sys.argv[1])

        # WARNING: Naive date time error while time zone support is active
        # (scrape_time does not have any time zone specified)
        posts_df.apply(
            save_post_from_row,
            axis=1
        )
        print("There are {} posts in DB".format(Post.objects.count()))

    else:
        print("Please, provide Posts file path")
