from django.shortcuts import render, get_object_or_404
from django.http import Http404
from django.forms.models import model_to_dict
from django.forms import ModelForm
from .collaborative_filter import update_filter

# Create your views here.

from .models import *


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
    except Page.DoesNotExist:
        raise Http404("Post does not exist")

    return render(request, 'recsys/page_detail.html', context)


def user_list(request):
    all_user_tests = UserTest.objects.all()[:100]
    context = {'user_list': all_user_tests}
    return render(request, 'recsys/user_list.html', context)


def user_detail(request, user_id):
    try:
        user = UserTest.objects.get(id=user_id)
        comments = Comment.objects.filter(from_id=user_id)
        # All comments that this user wrote
        context = {'comment_list': comments, 'user_name': user.name}
    except UserTest.DoesNotExist:
        raise Http404("Post does not exist")

    return render(request, 'recsys/user_detail.html', context)


def comment_list(request):
    pass


def comment_detail(request, comment_id):
    pass

# Recommendation using item-item collaborative filtering
def recommendation_CL(request):
    user_id = 10155675667923755 #remove this later?
    user_comments = Comment.objects.filter(from_id=user_id).order_by('created_time')
    user_posts = list(map(lambda x: x.post_id.id, user_comments))
    latest_post_id = user_posts[0]
    latest_post = Post.objects.get(id=latest_post_id)
    print(latest_post)

    try:
        similarities = CosineSimilarity.objects.get(source_id=latest_post) \
                    .exclude(target_id__in=user_posts) \
                    .order_by('similarity')
    except:
        update_filter()
        similarities = CosineSimilarity.objects.get(source_id=latest_post) \
                    .exclude(target_i__in=user_posts) \
                    .order_by('similarity')
    similar_post_ids = set(map(lambda x: x.target_id.id, similarities))
    similar_posts = Post.objects.filter(id__in=similar_post_ids)

    context = {'user_name': user.name, 'latest_post_list': similar_posts}
    return render(
        request,
        'recsys/recommendation_CL.html',
        context
    )
