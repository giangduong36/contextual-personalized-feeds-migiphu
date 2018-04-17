from django.shortcuts import render, get_object_or_404
from django.http import Http404
from django.forms.models import model_to_dict
from django.forms import ModelForm


# Create your views here.

from .models import Post, Page, Comment


def post_list(request):
    latest_post_list = Post.objects.order_by('created_time')[:50]
    context = {'latest_post_list': latest_post_list}
    return render(request, 'recsys/post_list.html', context)


def post_detail(request, post_id):
    class PostForm(ModelForm):
        class Meta:
            model = Post
            fields = '__all__'
    try:
        post = Post.objects.get(id=post_id)
        form = PostForm(instance=post)
        comments = Comment.objects.filter(post_id = post)
    except Post.DoesNotExist:
        raise Http404("Post does not exist")

    return render(request, 'recsys/post_detail.html', {'form': form, 'comments': comments})


def page_list(request):
    latest_page_list = Page.objects.all()
    context = {'latest_page_list': latest_page_list}
    return render(request, 'recsys/page_list.html', context)


def page_detail(request, page_id):
    try:
        page = Page.objects.get(id=page_id)
        posts_of_page = Post.objects.filter(page_id=page)   # All posts that belong to this page
        context = {'latest_post_list': posts_of_page, 'page_name': page.name}
    except posts_of_page.DoesNotExist:
        raise Http404("Post does not exist")

    return render(request, 'recsys/page_detail.html', context)


def comment_list(request):
    pass


def comment_detail(request, comment_id):
    pass