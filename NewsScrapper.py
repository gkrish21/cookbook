# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:40:38 2019

@author: GopalKrishna
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import pdb

import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer

from contractions import CONTRACTION_MAP
import unicodedata
nlp = spacy.load('en_core_web_md', parse=True, tag=True, entity=True)

#nlp_vec = spacy.load('en_vecs', parse = True, tag=True, #entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


seed_urls1 = {'WFC Stock': 'https://newsapi.org/v2/everything?q=WFC Stock&from=2019-10-15&page=1&pageSize=100&apiKey=0b1a54f7933c48ce8bf942a78e95ec9b',
             'Wells Fargo': 'https://newsapi.org/v2/everything?q=Wells Fargo&from=2019-10-15&page=1&pageSize=50&apiKey=0b1a54f7933c48ce8bf942a78e95ec9b',
             'Sports':'https://newsapi.org/v2/everything?q=Sports&from=2019-10-15&page=1&pageSize=100&apiKey=0b1a54f7933c48ce8bf942a78e95ec9b',
             'Weather': 'https://newsapi.org/v2/everything?q=Weather&from=2019-10-15&page=1&pageSize=100&apiKey=0b1a54f7933c48ce8bf942a78e95ec9b',
             'Technology':'https://newsapi.org/v2/everything?q=Technology&from=2019-10-15&page=1&pageSize=100&apiKey=0b1a54f7933c48ce8bf942a78e95ec9b',
             'Politics':'https://newsapi.org/v2/everything?q=Politics&from=2019-10-15&page=1&pageSize=100&apiKey=0b1a54f7933c48ce8bf942a78e95ec9b'}



mw_url = 'https://www.marketwatch.com/latest-news'

def build_df_from_market_watch(mw_url):
    data = requests.get(mw_url)
    soup = BeautifulSoup(data.content, 'html.parser')
    news_data = []
    news_articles = []

    for content in soup.find_all('div', class_=["article__content"]):
        news_articles = {}
        if content.find('h3', attrs={"class": "article__headline"}):
            news_articles['headline'] = content.find('h3', attrs={"class": "article__headline"}).string
        if content.find('p', attrs={"class": "article__summary"}):
            news_articles['summary'] = content.find('p', attrs={"class": "article__summary"}).string
        if content.find('span', attrs={"class": "ticker__symbol"}):
            news_articles['ticker'] = content.find('span', attrs={"class": "ticker__symbol"}).string
        if content.find('span', attrs={"class": "article__timestamp"}):
            news_articles['timestamp'] = content.find('span', attrs={"class": "article__timestamp"}).string
        if content.find('span', attrs={"class": "article__author"}):
            news_articles['author'] = content.find('span', attrs={"class": "article__author"}).string
        news_data.append(news_articles)
        
    df =  pd.DataFrame(news_data)
    return df


def build_df_from_google_news(seed_urls):
    news_data = []
    for category in seed_urls:
        url = seed_urls[category]
        data = requests.get(url).json()
        df = pd.DataFrame(data['articles'])
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        df['category'] = category
        news_data.append(df)
    
    news_df = pd.concat(news_data, ignore_index=True)
    news_df.rename(columns={"title": "headline", "description": "summary"}, inplace=True)
    return news_df
    

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True, remove_digits=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        
        if type(doc) is not str:
            normalized_corpus.append('')
            continue
        
        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus

if __name__ == '__main__':
    print('Hello World');
    
    # get news data from Market watch
    #news_df = build_df_from_market_watch(mw_url)
    
    # get news from google api
    news_df = build_df_from_google_news(seed_urls)
    
    # combining headline and article text
    news_df['full_text'] = news_df["headline"].map(str)+ '. ' + news_df["summary"]

    # pre-process text and store the same
    news_df['clean_text'] = normalize_corpus(news_df['full_text'])
    norm_corpus = list(news_df['clean_text'])
    
    # save the processed news data to csv file
    news_df.to_csv('news_data.csv')

    # show a sample news article
    print(news_df.iloc[0][['full_text', 'clean_text']].to_dict())