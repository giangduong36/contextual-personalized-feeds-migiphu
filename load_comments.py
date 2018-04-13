import sys, os
import pandas as pd
import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "recsys440.settings")

import django

django.setup()

from recsys.models import Comment, Page, Post
from django.utils import dateparse, timezone

def save_comment_from_row(comment_row):
    comment = Comment()
    comment.created_time = comment_row[0]
    try:
        comment.from_id = Page.objects.get(id=comment_row[1])
    except Page.DoesNotExist:
        comment.from_id = None

    # comment.from_id = Page.objects.get(id=comment_row[1])
    comment.from_name = comment_row[2]
    comment.message = comment_row[3]
    # comment.post_id = Post.objects.get(post_id=comment_row[4])
    try:
        comment.post_id = Page.objects.get(id=comment_row[4])
    except Page.DoesNotExist:
        comment.post_id = None

    comment.save()


def delete_db():
    print('truncate db')
    Comment.objects.all().delete()
    print('finished truncate db')


if __name__ == "__main__":

    if len(sys.argv) == 2:

        delete_db()
        print("Reading from file " + str(sys.argv[1]))
        comments_df = pd.read_csv(sys.argv[1])

        for col in comments_df.columns.tolist():
            print(col, comments_df[col].isnull().sum())

        print([dateparse.parse_datetime(i) for i in comments_df['created_time']])
        # print(list(set([type(i) for i in comments_df['created_time']])))

        # WARNING: Naive date time error while time zone support is active
        # (scrape_time does not have any time zone specified)
        # comments_df.apply(
        #     save_comment_from_row,
        #     axis=1
        # )

        print("There are {} comments in DB".format(Comment.objects.count()))

    else:
        print("Please, provide Comments file path")