# A Personalized News Feed

Personalized recommendations are ubiquitous on websites and influence the kind of information we receive.
We were interested in understanding, applying and comparing different recommender systems.
We were particularly interested in contextual bandit, an online-learning algorithm, that adapts to users’ activities to improve its recommendations.
This project consists of three main components:

+ A personalized news feed built with Django featuring news articles
+ Recommendations were created using: Tf-idf, Doc2Vec, Item-item Collaborative Filtering and an ensemble method using Learning to Rank
+ Contextual bandit recommender was trained as a proof of concept.

[[https://github.com/giangduong36/contextual-personalized-feeds-migiphu/blob/master/image/homepage.png]][home page]

!(contextual-personalized-feeds-migiphu/image/userpage.png)[list of users]

!(contextual-personalized-feeds-migiphu/image/userhomepage.png)[viewing a specific user]

!(contextual-personalized-feeds-migiphu/image/recommendationpage.png)[viewing recommendations]

### About the data:

Our news feed is built on a
[Facebook dataset](http://www.jbencina.com/blog/2017/07/14/facebook-news-dataset-1000k-comments-20k-posts/) collected
by John Bencina. It contains approximately 20,000 posts from 83 various news sources and a total of 1,000,000 comments from 462,431 users. Following are the fields in each data table:

+ Posts: created time, message, link, number of reactions and shares
+ News pages: name and id of news source
+ Comments: created time, id of parent post, name and id of user, message

## Getting Started

You can get a copy of the project and run it on your local server following
these instructions.

### Prerequisite

`Django - 2.0.4`

`Django boostrap3 - 9.1.0`

`numpy, scipy, sklearn, gensim`


### Installation

- Clone and do:

`python manage.py makemigrations recsys`

`python manage.py migrate`

- To run the server:

`python manage.py runserver 0.0.0.0:8000`

- Copy four data files linked below into the `data` folder (which is currently empty)

Copy and paste the [db.sqlite3](https://1fichier.com/?2r7bxhbnam) file in Drive to the repository
OR populating data yourself (will take a long time)

`python load_pages.py [data/fb_news_pagenames.csv](https://1fichier.com/?7nskh43qry)`

`python load_posts.py [data/fb_news_posts_20K.csv](https://1fichier.com/?vm3o938k7w)`

`python load_comments.py [data/fb_news_comments_1000k_cleaned.csv](https://1fichier.com/?o53ktx5a7o)`

`python load_users.py data/fb_news_comments_1000k_cleaned.csv`

- Open up a web browser and go to http://0.0.0.0:8000/recsys to see userhomepage

- Admin page with the whole datasets: http://0.0.0.0:8000/admin/

Username: comp440

Password: Hello321


## Contributors:

Giang Duong, Milo Beyene, Phuc Nguyen
