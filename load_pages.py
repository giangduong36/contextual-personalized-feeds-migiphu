import sys, os
import pandas as pd
import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "recsys440.settings")

import django

django.setup()

from reviews.models import Page, Post


def save_post_from_row(page_row):
    page = Page()
    page.name = page_row[0]
    page.id = page_row[1]
    page.save()


if __name__ == "__main__":

    if len(sys.argv) == 2:
        print("Reading from file " + str(sys.argv[1]))
        reviews_df = pd.read_csv(sys.argv[1])
        print(reviews_df.head())

        reviews_df.apply(
            save_post_from_row,
            axis=1
        )

        print("There are {} reviews in DB".format(Page.objects.count()))

    else:
        print("Please, provide Reviews file path")