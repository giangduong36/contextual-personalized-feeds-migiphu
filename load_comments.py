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
    comment.from_id = comment_row[1]
    comment.from_name = comment_row[2]
    comment.message = comment_row[3]

    try:
        comment.post_id = Post.objects.get(id=comment_row[4])
    except Post.DoesNotExist:
        comment.post_id = None

    comment.save()


def delete_db():
    print('truncate db')
    Comment.objects.all().delete()
    print('finished truncate db')


def is_valid_date(str):
    return dateparse.parse_datetime(str) is not None


def preprocess(path):
    # Clean comments dataset
    with open(path, 'r') as f:
        lines = f.read().splitlines()

    processed = []
    buffer = ""
    for line in lines:
        date = line.partition(",")[0]
        if dateparse.parse_datetime(date) is not None:
            if buffer != "":
                processed.append(buffer)
                buffer = ""
        buffer += line.replace('\n','')

    print("Length of dataset: ", len(processed))

    with open('data/fb_news_comments_1000k_cleaned.csv', 'w') as the_file:
        for i in processed:
            the_file.write(i + '\n')


if __name__ == "__main__":

    if len(sys.argv) == 2:

        delete_db()
        print("Reading from file " + str(sys.argv[1]))
        comments_df = pd.read_csv(sys.argv[1])
        comments_df.drop_duplicates(inplace=True)   # Comments data set have duplicates
        print(comments_df.info())

        # preprocess(str(sys.argv[1])) # Use for the ORIGINAL fb_news_comments_1000k.csv

        # WARNING: Naive date time error while time zone support is active
        # (scrape_time does not have any time zone specified)
        comments_df.apply(
            save_comment_from_row,
            axis=1
        )

        print("There are {} comments in DB".format(Comment.objects.count()))

    else:
        print("Please, provide Comments file path")
