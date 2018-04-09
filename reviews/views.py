from django.shortcuts import render, get_object_or_404

# Create your views here.

from .models import Post, Page


def post_list(request):
    latest_review_list = Post.objects.all()
    context = {'latest_review_list':latest_review_list}
    return render(request, 'reviews/post_list.html', context)


def post_detail(request, post_id):
    post = get_object_or_404(Post, pk=post_id)
    return render(request, 'reviews/post_detail.html', {'post': post})


def page_list(request):
    latest_review_list = Page.objects.all()
    context = {'latest_review_list':latest_review_list}
    return render(request, 'reviews/page_list.html', context)


def page_detail(request, page_id):
    page = get_object_or_404(Page, pk=page_id)
    return render(request, 'reviews/page_detail.html', {'page': page})

# def wine_list(request):
#     wine_list = Wine.objects.order_by('-name')
#     context = {'wine_list':wine_list}
#     return render(request, 'reviews/wine_list.html', context)
#
#
# def wine_detail(request, wine_id):
#     wine = get_object_or_404(Wine, pk=wine_id)
#     return render(request, 'reviews/wine_detail.html', {'wine': wine})
