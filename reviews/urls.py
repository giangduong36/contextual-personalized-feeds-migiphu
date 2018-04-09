from django.conf.urls import url
from django.urls import path

from . import views

urlpatterns = [
    # ex: /
    url(r'^$', views.post_list, name='post_list'),
    # ex: /review/5/
    url(r'^post/(?P<post_id>[0-9]+)/$', views.post_detail, name='post_detail'),
    # ex: /wine/
    url(r'^page', views.page_list, name='page_list'),
    # ex: /wine/5/
    url(r'^page/(?P<page_id>[0-9]+)/$', views.page_detail, name='page_detail'),

    # path('', views.post_list, name='post_list'),
    # # ex: /polls/5/
    # path('^post/<int:review_id>/', views.post_detail, name='post_detail'),
    # # ex: /polls/5/results/
    # path('<int:question_id>/results/', views.results, name='results'),
    # # ex: /polls/5/vote/
    # path('<int:question_id>/vote/', views.vote, name='vote'),

]
