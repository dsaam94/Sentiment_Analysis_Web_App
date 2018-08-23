# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 12:36:27 2018

@author: Ali Asghar Marvi
"""

from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='template')

@app.route('/')
def home():
	return render_template('home.htm')

        
cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(
                os.path.join(cur_dir, 
                'movieclassifier\pkl_objects', 
                'stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
                   + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

@app.route('/predict_sent',methods=['POST','GET'])
def predict_sent():
    if request.method=='POST':
        result=request.form
        sample_review = result['review']
    clf = pickle.load(open(os.path.join('movieclassifier\pkl_objects', 'classifier.plk'), 'rb'))
    sample_review = sample_review.split(' ')
    X = vect.transform(sample_review)
    label = {0:'negative', 1:'positive'}
    k = ('Prediction: %s\nProbability: %.2f%%' %\
      (label[clf.predict(X)[0]], 
       np.max(clf.predict_proba(X))*100))
    return render_template('predict_sent.htm',prediction=k)


if __name__ == '__main__':
	app.debug = True
	app.run()





