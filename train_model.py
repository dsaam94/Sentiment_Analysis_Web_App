# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 03:39:04 2018

@author: Ali Asghar Marvi
"""

import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import pickle
import os 

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


clf = SGDClassifier(loss='log', random_state=1, n_iter=1)

porter = PorterStemmer()
stop = stopwords.words('english')

dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
  os.makedirs(dest)
  
pickle.dump(stop, 
            open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
            protocol = 4)

def stream_docs(path):
  with open(path, 'r', encoding='utf-8') as csv:
      next(csv)  # skip header
      for line in csv:
          text, label = line[:-3], int(line[-2])
          yield text, label
            
doc_stream = stream_docs(path='movie_data.csv')

def get_minibatch(doc_stream, size):
  docs, y = [], []
  try:
      for _ in range(size):
          text, label = next(doc_stream)
          docs.append(text)
          y.append(label)
  except StopIteration:
      return None, None
  return docs, y
 
def tokenizer(text):
  text = re.sub('<[^>]*>', '', text)
  emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
  text = re.sub('[\W]+', ' ', text.lower()) +\
      ' '.join(emoticons).replace('-', '')
  tokenized = [w for w in word_tokenize(text) if w not in stop]
#  tokenized = [porter.stem(w) for w in tokenized ]
  return tokenized
  
vect = HashingVectorizer(decode_error='ignore', 
                       n_features=2**21,
                       preprocessor=None, 
                       tokenizer=tokenizer)
def train():

    classes = np.array([0, 1])
    print('Iterating with 45 mini batches of 1000 each')
    for i in range(45):
      print('Training on ' + str(i+1)+'th ' + 'iteration')  
      X_train, y_train = get_minibatch(doc_stream, size=1000)
      if not X_train:
            break
      X_train = vect.transform(X_train)
      clf.partial_fit(X_train, y_train, classes=classes)
      print('Accuracy: ' + str(clf.score(X_train, y_train)*100 ) + '%')
    print('Training completed')
    pickle.dump(clf, open(os.path.join(dest, 'classifier.plk'), 'wb'))
    print('Model Saved to ../movieclassifier/pk1_objects')
    
def test():
    print('Test model for accuracy with last 5000 rows')
    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print('Accuracy: %.3f' % (clf.score(X_test, y_test)*100))
    
train()
test()