# A Personalized News Feed

Personalized recommendations are ubiquitous on websites and influence the kind of information we receive.
We were interested in understanding, applying and comparing different recommender systems.
We were particularly interested in contextual bandit, an online-learning algorithm, that adapts to users’ activities to improve its recommendations.
This project consists of three main components:

+ A personalized news feed built with Django featuring news articles
+ Recommendations were created using: Tf-idf, Doc2Vec, Item-item Collaborative Filtering and an ensemble method using Learning to Rank
+ Contextual bandit recommender (LinUCB algorithm) was trained as a proof of concept.

Here are some screen shots of the news feed:

The home page

![home page](https://github.com/giangduong36/contextual-personalized-feeds-migiphu/blob/master/image/homepage.png?raw=true)

Display of list of users (mainly for testing purposes)

![list of users](https://github.com/giangduong36/contextual-personalized-feeds-migiphu/blob/master/image/userpage.png?raw=true)

Clicking on a specific user shows posts they've read in the past, and options to view recommendations

![view of specific user](https://github.com/giangduong36/contextual-personalized-feeds-migiphu/blob/master/image/userhomepage.png?raw=true)

View of recommendations

![view of recommendations](https://github.com/giangduong36/contextual-personalized-feeds-migiphu/blob/master/image/recommendationpage.png?raw=true)

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

Please refer to `requirements.txt` file. Some of the most important packages are:

`Django 2.0.4`

`Django boostrap3 9.1.0`

`numpy, scipy, sklearn, gensim`


### Installation

- Clone the repository and do:

`python manage.py makemigrations recsys`

`python manage.py migrate`

- To run the server:

`python manage.py runserver 0.0.0.0:8000`

- Copy four data files linked below into the `data` folder (which is currently empty)

Copy and paste the [db.sqlite3](https://1fichier.com/?2r7bxhbnam) file in Drive to the repository
OR populating data yourself with the following 4 commands(will take a long time)

`python load_pages.py data/fb_news_pagenames.csv` [download](https://1fichier.com/?7nskh43qry)

`python load_posts.py data/fb_news_posts_20K.csv` [download](https://1fichier.com/?vm3o938k7w)

`python load_comments.py data/fb_news_comments_1000k_cleaned.csv` [download](https://1fichier.com/?o53ktx5a7o)

`python load_users.py data/fb_news_comments_1000k_cleaned.csv`

- Open up a web browser and go to http://0.0.0.0:8000/recsys to see userhomepage

- Admin page with the whole datasets: http://0.0.0.0:8000/admin/

Username: comp440

Password: Hello321

## Run contextual bandit LinUCB

- Populate necessary data files to `data` folder: [download](https://1fichier.com/?qdeboefe1i)
- Go to `recsys/recommender/` and run `python ./contextual_bandit_simulate.py` 
- To test LinUCB on a sample dataset: do `run_sample_dataset()` in the main function of `contextual_bandit_simulate.py` file.
- To run LinUCB on the Facebook News Dataset: using the example code, create a new sample with a size of your choice and do `run_generated_dataset`.
- The result image for the cumulative click through rate will be saved in the same folder.

## Run Learning-To-Rank ensemble recommender
- To run the Learning-To-Rank ensemble on existing train, validation, and test sets of size 5000 most active users, go to `recsys/recommender/` and run `python ./learning_to_rank_ensemble.py`.
- To run the Learning-To-Rank ensemble on new train, validation, and test sets:
  * Run `create_dataset_for_learning_to_rank(n_users)`
  * Follow the [documentation from The  Lemur Project](https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/). 
  * Example command to train and validate on newly created data files: `java -jar RankLib.jar -train train1000.txt -validate validate1000.txt -ranker 4 -save model1000.txt -feature feature_list.txt -r 1 -i 25 -tolerance 0.001 -silent -metric2t NDCG@10 -norm zscore`
  * Example command to test on newly created data files: `java -jar RankLib.jar -load model1000.txt -rank test1000.txt -score score1000.txt -feature feature_list.txt -norm zscore`

## Contributors:

Giang Duong, Milo Beyene, Phuc Nguyen

## Acknowledgement:

Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. Proceedings of the 19th International Conference on World Wide Web - WWW 10. doi:10.1145/1772690.1772758

Learn to Rank: Implementation RankLib from [the LEMUR project](https://sourceforge.net/p/lemur/wiki/Home/)

