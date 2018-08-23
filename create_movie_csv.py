# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 23:41:53 2018

@author: Ali Asghar Marvi
"""

import pandas as pd
import os
import os.path as path
import pyprind
import numpy as np
import tarfile
from six.moves import urllib
import sys
import requests
#path = 'F:\ML Web App\data\\aclImdb\\test'

DATA_DOWNLOAD_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_PATH = os.path.join("data")

def download_file(data_download_url, data_path):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    with open('../data/aclImdb_v1.tar.gz', 'wb') as f:
        response = requests.get(data_download_url, stream=True)
        total = response.headers.get('content-length')
        
        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')
    
    
def fetch__data(data_download_url, data_path):
    print('Downloading aclImdb_v1.tar.gz from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz....')
    download_file(data_download_url, data_path)
    print('aclImdb_v1.tar.gz Downloaded')
    if os.path.isfile("../data/aclImdb_v1.tar.gz"):
        print('Extracting aclImdb_v1.tar.gz......') 
        print('Please wait it should take approximately 20 mins......') 
        reviews_tgz = tarfile.open("../data/aclImdb_v1.tar.gz")
        reviews_tgz.extractall(path=DATA_PATH)
        reviews_tgz.close()
    print('Data Extracted in ' + 'data '+ 'folder') 



def create_df():

    basepath =  path.abspath(path.join(os.getcwd(),"../ML Web App/data/aclImdb"))
    print (basepath)
    labels = {'pos': 1, 'neg': 0}
    pbar = pyprind.ProgBar(50000)
    df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            bpath = os.path.join(basepath, s, l)
            for file in os.listdir(bpath):
                with open(os.path.join(bpath, file), 
                          'r', encoding='utf-8') as infile:
                    txt = infile.read()
                df = df.append([[txt, labels[l]]], 
                               ignore_index=True)
                pbar.update()
    df.columns = ['review', 'sentiment']
#    
#    print(df.head())
    return df

def randomize_data_save_to_csv():
    print('Creating DataFrame for csv....')
    df = create_df()
    print('Finished Creating DataFrame....')
    np.random.seed(0)
    #randomize data frame
    print('Randomizing Rows....')
    df = df.reindex(np.random.permutation(df.index))
    #save data frame to a csv file called movie_data.csv
    print('Saving to movie_data.csv file....')
    df.to_csv('movie_data.csv', index = False, encoding = 'utf-8')
    print('File movie_data.csv created in main directory....')

fetch__data(DATA_DOWNLOAD_URL,DATA_PATH)        
randomize_data_save_to_csv()
