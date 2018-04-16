from django.conf.urls import url
from django.urls import path

from . import views

app_name = 'recsys'

urlpatterns = [
    path('', views.post_list, name='post_list'),
    path('post/', views.post_list, name='post_list'),
    # ex: /post/5/
    path('post/<post_id>/', views.post_detail, name='post_detail'),

    path('page/', views.page_list, name='page_list'),
    # ex: /page/5/
    path('page/<page_id>/', views.page_detail, name='page_detail'),
]
