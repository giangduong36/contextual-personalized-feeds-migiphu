from django.db import models
import numpy as np


# Create your models here.


class Page(models.Model):
    name = models.CharField(max_length=200, unique=True, blank=True, null=True)
    id = models.CharField(max_length=50, primary_key=True)

    def __unicode__(self):
        return self.name


class Post(models.Model):
    # page = models.ForeignKey(Page, on_delete=models.DO_NOTHING)
    # No need _id for foreignkey
    page_id = models.ForeignKey(Page, on_delete=models.CASCADE, blank=True, null=True)
    id = models.CharField(max_length=100, primary_key=True)
    created_time = models.DateTimeField('created time', blank=True, null=True)
    description = models.CharField(max_length=200, blank=True, null=True)
    link = models.CharField(max_length=300, blank=True, null=True)
    message = models.CharField(max_length=2000, blank=True, null=True)
    react_angry = models.IntegerField(blank=True, null=True)
    react_haha = models.IntegerField(blank=True, null=True)
    react_like = models.IntegerField(blank=True, null=True)
    react_love = models.IntegerField(blank=True, null=True)
    react_sad = models.IntegerField(blank=True, null=True)
    react_wow = models.IntegerField(blank=True, null=True)
    scrape_time = models.DateTimeField('scrape time', blank=True, null=True)
    shares = models.IntegerField(blank=True, null=True)


class Comment(models.Model):
    created_time = models.DateTimeField('created time', blank=True, null=True)
    from_id = models.CharField(max_length=200, blank=True, null=True)  # user's id
    from_name = models.CharField(max_length=200, blank=True, null=True)  # user's public name
    message = models.CharField(max_length=2000, blank=True, null=True)
    post_id = models.ForeignKey(Post, on_delete=models.CASCADE, blank=True, null=True)


# A convenient and quick model to represent users from the existing database.
# Creating user authentication for all users in the database is time-consuming
class UserTest(models.Model):
    id = models.CharField(max_length=100, primary_key=True)
    name = models.CharField(max_length=200, blank=True, null=True)  # user's public name


class SuggestPost(models.Model):
    pass