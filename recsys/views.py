from django.shortcuts import render, get_object_or_404
from django.http import Http404


# Create your views here.

from .models import Post, Page


def post_list(request):
    latest_post_list = Post.objects.order_by('created_time')[:50]
    context = {'latest_post_list': latest_post_list}
    return render(request, 'recsys/post_list.html', context)


def post_detail(request, post_id):
    try:
        post = Post.objects.get(id=post_id)
    except Post.DoesNotExist:
        raise Http404("Post does not exist")

    return render(request, 'recsys/post_detail.html', {'post': post})


def page_list(request):
    latest_page_list = Page.objects.all()
    context = {'latest_page_list':latest_page_list}
    return render(request, 'recsys/page_list.html', context)


def page_detail(request, page_id):
    page = get_object_or_404(Page, pk=page_id)
    return render(request, 'recsys/page_detail.html', {'page': page})


def comment_list(request):
    pass


def comment_detail(request, comment_id):
    pass