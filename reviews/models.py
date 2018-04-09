from django.db import models
import numpy as np

# Create your models here.

class Page(models.Model):
    name = models.CharField(max_length=200)
    id = models.CharField(max_length=50, primary_key=True)

    def __unicode__(self):
        return self.name


class Post(models.Model):
    page = models.ForeignKey(Page, on_delete=models.DO_NOTHING)
    created_time = models.DateTimeField('created time')
    description = models.CharField(max_length=200)
    link = models.CharField(max_length=300)
    message = models.CharField(max_length=1000)
    # page_id
    # post_id
    react_angry = models.IntegerField()
    react_haha = models.IntegerField()
    react_like = models.IntegerField()
    react_love = models.IntegerField()
    react_sad = models.IntegerField()
    react_wow = models.IntegerField()
    scrape_time = models.DateTimeField('scrape time')
    shares = models.IntegerField()
