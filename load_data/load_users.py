import os
import sys

import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "recsys440.settings")

import django

django.setup()

from django.contrib.auth.models import User

from recsys.models import UserTest


# Create User Authentication with User ID and Name from dataset
# http://www.jbencina.com/blog/2017/07/14/facebook-news-dataset-1000k-comments-20k-posts/
def create_user(comment_row):
    user_id = str(comment_row[1])

    try:
        user = User.objects.get(username=user_id)
    except User.DoesNotExist:
        user = User.objects.create_user(
            username=user_id,  # Save the original id of user as the username
            first_name=comment_row[2],  # Get the name of the user
            last_name=user_id,  # Save the original id of user as the last name for easy retrieval
            password="Hello321",  # Default password
        )
        user.save()


# Load users as a Model, not a User Authentication (much faster, for testing)
def create_user_temp(comment_row):
    try:
        UserTest.objects.get(id=comment_row[1])
    except UserTest.DoesNotExist:
        user = UserTest()
        user.id = comment_row[1]
        user.name = comment_row[2]
        user.save()


def delete_db():
    print('truncate db')
    UserTest.objects.all().delete()
    print('finished truncate db')


if __name__ == "__main__":

    if len(sys.argv) == 2:

        delete_db()
        print("Reading from file " + str(sys.argv[1]))
        comments_df = pd.read_csv(sys.argv[1])
        comments_df.drop_duplicates(inplace=True)  # Comments data set have duplicates

        # WARNING: Naive date time error while time zone support is active
        # (scrape_time does not have any time zone specified)

        comments_df.apply(
            create_user_temp,
            axis=1
        )

        print("There are {} users in DB".format(UserTest.objects.count()))

    else:
        print("Please, provide Comments file path")
