# Based off of:
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-IMDB.ipynb
# https://www.kaggle.com/tj2552/sentiment-classification-in-5-classes-doc2vec?scriptVersionId=373473

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
import pandas as pd

lmtzr = WordNetLemmatizer()
w = re.compile("\w+", re.I)

tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))

def nlp_clean(data): # remove new lines
   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data

def label_sentences(df):
    labeled_sentences = []
    for index, datapoint in df.iterrows():
        tokenized_words = re.findall(w, datapoint["message"])
        labeled_sentences.append(TaggedDocument(words=tokenized_words, tags=[index]))
    return labeled_sentences

def train_doc2vec_model(labeled_sentences, max_epochs = 100, vec_size = 20, alpha = 0.025):
    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.025,
                    min_count=1,
                    dm=1)
    model.build_vocab(labeled_sentences)
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(labeled_sentences,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    return model

def find_most_similar_doc(df, model):
    similar = []
    for i in range(0, df.shape[0]):
        sim_index = model.docvecs.most_similar([model.docvecs[i]])[1][0]
        postid = df.ix[sim_index]['post_id']
        similar.append(postid)
    df['most_similar'] = similar
    df.to_csv(path_or_buf='../../data/fb_news_posts_20K_doc2vec.csv',
                         index=False,
                         columns=['post_id', 'most_similar'])
    return df


df = pd.read_csv('../../data/fb_news_posts_20K.csv')[['post_id', 'message']]
df.fillna('', inplace=True)

sen = label_sentences(df)
# model = train_doc2vec_model(sen)
# model.save("d2v.model")
# print("Model Saved")

# Load Model instead...
model = Doc2Vec.load("d2v.model")

find_most_similar_doc(df, model)
print(df.head(5))