from django.db import models
import numpy as np

# Create your models here.


class Page(models.Model):
    name = models.CharField(max_length=200, unique=True)
    id = models.CharField(max_length=50, primary_key=True)

    def __unicode__(self):
        return self.name


class Post(models.Model):
    # page = models.ForeignKey(Page, on_delete=models.DO_NOTHING)
    page_id = models.ForeignKey(Page, on_delete=models.CASCADE)
    post_id = models.CharField(max_length=100, primary_key=True)
    created_time = models.DateTimeField('created time')
    description = models.CharField(max_length=200)
    link = models.CharField(max_length=300)
    message = models.CharField(max_length=2000)
    react_angry = models.IntegerField()
    react_haha = models.IntegerField()
    react_like = models.IntegerField()
    react_love = models.IntegerField()
    react_sad = models.IntegerField()
    react_wow = models.IntegerField()
    scrape_time = models.DateTimeField('scrape time')
    shares = models.IntegerField()


class Comment(models.Model):
    created_time = models.DateTimeField('created time')
    from_id = models.ForeignKey(Page, to_field='id', on_delete=models.CASCADE)
    from_name = from_id.name
    message = models.CharField(max_length=2000)
    post_id = models.ForeignKey(Post, on_delete=models.CASCADE)
