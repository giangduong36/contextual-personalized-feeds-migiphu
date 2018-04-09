from django.contrib import admin

from .models import Page, Post


class ReviewAdmin(admin.ModelAdmin):
    model = Post
    list_display = ('created_time', 'link', 'message')
    list_filter = ['message']
    search_fields = ['message']


admin.site.register(Page)
admin.site.register(Post, ReviewAdmin)