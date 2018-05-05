from django.shortcuts import render, get_object_or_404
from django.http import Http404
from django.db.models import Count
from django.forms import ModelForm
from .recommender.collaborative_filter import update_filter
import pandas as pd
import sys
from django.db.models import Q
import re
from django import template

register = template.Library()

# Create your views here.

from .models import *

"""View a list of all pages"""


def page_list(request):
    strings = ['recsys/PusheenPack/cake.png', 'recsys/PusheenPack/art.png', 'recsys/PusheenPack/cookies.png',
               'recsys/PusheenPack/DJ.png', 'recsys/PusheenPack/Games.png', 'recsys/PusheenPack/Music.png',
               'recsys/PusheenPack/pizza.png', 'recsys/PusheenPack/Pusheen.png', 'recsys/PusheenPack/R2D2.png',
               'recsys/PusheenPack/Sailor Moon.png', 'recsys/PusheenPack/school.png', 'recsys/PusheenPack/Sushi.png',
               'recsys/PusheenPack/Unicorn.png']
    latest_page_list = Page.objects.all()
    context = {'latest_page_list': latest_page_list, 'images': strings}
    return render(request, 'recsys/page_list.html', context)


"""View all posts of a page"""


def page_detail(request, page_id):
    try:
        page = Page.objects.get(id=page_id)
        posts_of_page = Post.objects.filter(page_id=page)  # All posts that belong to this page
        context = {'latest_post_list': posts_of_page, 'page_name': page.name}
    except Page.DoesNotExist:
        raise Http404("Post does not exist")

    return render(request, 'recsys/page_detail.html', context)


"""View a list of all posts"""


def post_list(request):
    latest_post_list = Post.objects.order_by('created_time')[:50]
    context = {'latest_post_list': latest_post_list}
    return render(request, 'recsys/post_list.html', context)


"""View information and comments of a post"""


def post_detail(request, post_id):
    class PostForm(ModelForm):
        class Meta:
            model = Post
            fields = '__all__'

    try:
        post = Post.objects.get(id=post_id)
        form = PostForm(instance=post)
        comments = Comment.objects.filter(post_id=post)
    except Post.DoesNotExist:
        raise Http404("Post does not exist")

    return render(request, 'recsys/post_detail.html', {'form': form, 'comments': comments})


"""View a list of all users"""


def user_list(request):
    strings = ['recsys/PusheenPack/cake.png', 'recsys/PusheenPack/art.png', 'recsys/PusheenPack/cookies.png',
               'recsys/PusheenPack/DJ.png', 'recsys/PusheenPack/Games.png', 'recsys/PusheenPack/Music.png',
               'recsys/PusheenPack/pizza.png', 'recsys/PusheenPack/Pusheen.png', 'recsys/PusheenPack/R2D2.png',
               'recsys/PusheenPack/Sailor Moon.png', 'recsys/PusheenPack/school.png', 'recsys/PusheenPack/Sushi.png',
               'recsys/PusheenPack/Unicorn.png']

    all_user_tests = UserTest.objects.all()[:100]
    context = {'user_list': all_user_tests, 'images': strings}
    return render(request, 'recsys/user_list.html', context)


"""View all comments of a user"""


def user_detail(request, user_id):
    try:
        user = UserTest.objects.get(id=user_id)
        comments = Comment.objects.filter(from_id=user_id)
        # All comments that this user wrote
        context = {'comment_list': comments,
                   'user_name': user.name,
                   'user_id': user_id}
    except UserTest.DoesNotExist:
        raise Http404("Post does not exist")

    return render(request, 'recsys/user_detail.html', context)


def user_recommended_post(request, user_id):
    user_comments = Comment.objects.filter(from_id=user_id)
    user_comments_post_id = set(map(lambda x: x.post_id.id, user_comments))
    tfidf_rec = pd.read_csv('data/fb_news_posts_20K_tfidf.csv')

    most_similar = tfidf_rec.loc[tfidf_rec['post_id'].isin(user_comments_post_id)]

    rec_posts_ids = []
    for i in most_similar['most_similar'].tolist():
        most_similar_ids = (i[1:-1].split(','))
        most_similar_ids = [i.replace(" ", "").replace('\'', "") for i in most_similar_ids]
        rec_posts_ids.extend(most_similar_ids)

    rec_posts = Post.objects.filter(pk__in=rec_posts_ids)
    context = {'rec_posts': rec_posts,
               'user_name': UserTest.objects.get(pk=user_id).name}
    return render(request, 'recsys/user_recommended_post.html', context)

def normalize_query(query_string,
    findterms=re.compile(r'"([^"]+)"|(\S+)').findall,
    normspace=re.compile(r'\s{2,}').sub):

    '''
    Splits the query string in invidual keywords, getting rid of unecessary spaces and grouping quoted words together.
    Example:
    >>> normalize_query('  some random  words "with   quotes  " and   spaces')
        ['some', 'random', 'words', 'with quotes', 'and', 'spaces']
    '''

    return [normspace(' ',(t[0] or t[1]).strip()) for t in findterms(query_string)]

def get_query(query_string, search_fields):

    '''
    Returns a query, that is a combination of Q objects.
    That combination aims to search keywords within a model by testing the given search fields.
    '''

    query = None # Query to search for every search term
    terms = normalize_query(query_string)
    for term in terms:
        or_query = None # Query to search for a given term in each field
        for field_name in search_fields:
            q = Q(**{"%s__startswith" % field_name: term})
            if or_query is None:
                or_query = q
            else:
                or_query = or_query | q
        if query is None:
            query = or_query
        else:
            query = query & or_query
    return query

def user_search(request):
    strings = ['recsys/PusheenPack/cake.png', 'recsys/PusheenPack/art.png', 'recsys/PusheenPack/cookies.png',
               'recsys/PusheenPack/DJ.png', 'recsys/PusheenPack/Games.png', 'recsys/PusheenPack/Music.png',
               'recsys/PusheenPack/pizza.png', 'recsys/PusheenPack/Pusheen.png', 'recsys/PusheenPack/R2D2.png',
               'recsys/PusheenPack/Sailor Moon.png', 'recsys/PusheenPack/school.png', 'recsys/PusheenPack/Sushi.png',
               'recsys/PusheenPack/Unicorn.png']
    all_user_tests = UserTest.objects.all()
    query_string = ''
    found_entries = None
    if ('q' in request.GET) and request.GET['q'].strip():
        query_string = request.GET['q']
        entry_query = get_query(query_string, ['id', 'name'])
        found_entries = all_user_tests.filter(entry_query) #.order_by('-something')

    context = {'query_string': query_string, 'entry_query': entry_query, 'found_entries': found_entries, 'images': strings}
    return render(request, 'recsys/user_search.html', context)


def comment_list(request):
    pass


def comment_detail(request, comment_id):
    pass


# Recommendation using Doc2Vec
def recommendation_d2v(request, user_id):
    user_comments = Comment.objects.filter(from_id=user_id)
    user_comments_post_id = set(map(lambda x: x.post_id.id, user_comments))
    d2v_rec = pd.read_csv('data/fb_news_posts_20K_doc2v.csv')

    most_similar = d2v_rec.loc[d2v_rec['post_id'].isin(user_comments_post_id)]

    rec_posts_ids = []
    for i in most_similar['most_similar'].tolist():
        most_similar_ids = (i[1:-1].split(','))
        most_similar_ids = [i.replace(" ", "").replace('\'', "") for i in most_similar_ids]
        rec_posts_ids.extend(most_similar_ids)

    rec_posts = Post.objects.filter(pk__in=rec_posts_ids)
    context = {'rec_posts': rec_posts,
               'user_name': UserTest.objects.get(pk=user_id).name}
    return render(request, 'recsys/user_recommended_post.html', context)


# Recommendation using item-item collaborative filtering
def recommendation_CF(request, user_id):

    '''
    Use item-item colloborative filtering to recommend posts to current user
    '''
    user_comments = Comment.objects.filter(from_id=user_id).order_by('created_time')
    user_posts_ids = list(map(lambda x: x.post_id.id, user_comments))
    latest_post_id = user_posts_ids[0]
    latest_post = Post.objects.get(id=latest_post_id)
    rec_posts = Post.objects.filter(id__in=user_posts_ids)

    comparable = CosineSimilarity.objects.filter(source_id=latest_post_id)
    if not comparable:
        # This should only run the first time this function is called
        # Periodic updates of the cosine similarity should be done offline so there isn't a lag
        # TODO: Replace update_filter() with a function that only updates the new posts that aren't in the database
        update_filter()
    similarities = CosineSimilarity.objects.filter(source_id=latest_post_id) \
                    .exclude(target_id__in=user_posts_ids).order_by('-similarity')
    rec_posts_ids = list(map(lambda x: x.target_id, similarities))
    rec_posts = []
    for id in rec_posts_ids:
        rec_posts.append(Post.objects.get(id=id))
    context = {'rec_posts': rec_posts,
            'user_name': UserTest.objects.get(id=user_id).name}
    return render(
        request,
        'recsys/user_recommended_post.html',
        context
    )
