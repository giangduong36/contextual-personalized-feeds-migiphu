from django.contrib import admin

from .models import *


class PostAdmin(admin.ModelAdmin):
    model = Post
    list_display = ('created_time', 'message', 'page_id', 'id', 'link', )
    list_filter = ['page_id']
    search_fields = ['page_id']


class PageAdmin(admin.ModelAdmin):
    model = Page
    list_display = ['name', 'id']


class CommentAdmin(admin.ModelAdmin):
    model = Comment
    list_display = ['created_time', 'post_id', 'message', 'from_id', 'from_name']
    list_filter = []


class UserTestAdmin(admin.ModelAdmin):
    model = Comment
    list_display = ['id', 'name']
    list_filter = []


class CosineSimilarityAdmin(admin.ModelAdmin):
    model = CosineSimilarity
    list_display = ['source_id', 'target_id', 'id']


admin.site.register(Page, PageAdmin)
admin.site.register(Post, PostAdmin)
admin.site.register(Comment, CommentAdmin)
admin.site.register(UserTest, UserTestAdmin)
admin.site.register(CosineSimilarity, CosineSimilarityAdmin)
