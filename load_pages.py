import sys, os
import pandas as pd
import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "recsys440.settings")

import django

django.setup()

from recsys.models import Page


def save_page_from_row(page_row):
    page = Page()
    page.name = page_row[0]
    page.id = page_row[1]
    page.save()


def delete_db():
    print('truncate db')
    Page.objects.all().delete()
    print('finished truncate db')


if __name__ == "__main__":

    if len(sys.argv) == 2:
        delete_db()
        print("Reading from file " + str(sys.argv[1]))
        pages_df = pd.read_csv(sys.argv[1])
        print(pages_df.head())

        pages_df.apply(
            save_page_from_row,
            axis=1
        )

        print("There are {} pages in DB".format(Page.objects.count()))

    else:
        print("Please, provide Pages file path")