from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
import joblib
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.text import Text
from nltk import pos_tag
from nltk.corpus import stopwords
import translators as ts
from multiprocessing import Pool
from tqdm import *
from nltk.corpus import wordnet as wn
import numpy as np
from scipy.stats import entropy
import math
post_qa_question = pd.read_excel('d:\.毕业论文\qa_system\post_qa_question_100.xlsx')

questions = []
answers = []
for i in range(len(post_qa_question)):
    questions.append(post_qa_question['Title'][i])
    answers.append(post_qa_question['Answer_Body'][i])
def syn_wordnet(word):
    # 通过WordNet获取词语同义词集的方法
    word_synset = set()
    # TODO 获取WordNet中的同义词集
    synsets = wn.synsets(word)  # word所在的词集列表
    for synset in synsets:
        words = synset.lemma_names()
        for word in words:
            word = word.replace('_', ' ')
            word_synset.add(word)
    return list(word_synset)

#对列表的元素进行替换
def list_replace(list,str1,str2):
    list_copy = list.copy()
    for i in range(len(list_copy)):
        if list_copy[i] == str1:
            list_copy[i] = str2
    return list_copy

imp_tag = ['NN','NNS','RB','VB','VBD','VBG','VBN','JJ','JJR','JJS']#重要的词性表
ban_word = ['do'] #禁止替换词汇，如替换会造成一些不好的后果
def synonym_replace_wordnet (text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    synonym_sentence = [] #目前的同义句
    #synonym_sentence用来存储当前的句子
    synonym_sentence.append(tokens)
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    for i in range(len(tags)):
        #tags为原始句子当前单词，若为重要标签
        if tags[i][1] in imp_tag and tags[i][0] not in ban_word:
            synonym_tmp = syn_wordnet(tags[i][0])
            synonym = []
            for j in range(len(synonym_tmp)):
                if pos_tag(word_tokenize(synonym_tmp[j]))[0][1] == tags[i][1]:
                    synonym.append(synonym_tmp[j])
            min_time = min(len(synonym),2)
            for j in range(min_time):
                #现在已经有重要词汇替换表了，再用一个for循环来扩充
                for k in range(len(synonym_sentence)):
                    list = list_replace(synonym_sentence[k],tokens[i],synonym[j])
                    synonym_sentence.append(list)
    return synonym_sentence

def tfidf_extractor(corpus, ngram_range=(1, 3)):
    vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True, use_idf=True, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    # print('####',features)
    return vectorizer, features


# # tfidf 特征
def conver2tfidf(data):
    new_data = []
    for q in data:
        new_data.append(q)
    tfidf_vectorizer, tfidf_X = tfidf_extractor(new_data)
    return tfidf_vectorizer, tfidf_X
tfidf_vectorizer, tfidf_X = conver2tfidf(questions)


        
#最大余弦距离
def idx_for_largest_cosine_sim(input, questions):
    list = []
    input = (input.toarray())[0]
    for question in questions:
        question = question.toarray()
        num = float(np.matmul(question, input))
        denom = np.linalg.norm(question) * np.linalg.norm(input)
        if denom ==0:
            cos = 0.0
        else:
            cos = num / denom
        list.append(cos)
    best_idx = list.index(max(list))
    return best_idx

# 返回最佳答案的索引
def answer_tfidf(input):
    bow = tfidf_vectorizer.transform([input])
    best_idx_cos = idx_for_largest_cosine_sim(bow, tfidf_X)
    return best_idx_cos

question_input = input()
questions_input = []

#通过回译来进行扩增
question1 = ts.translate_text(query_text = question_input, translator = 'bing',from_language= 'en',to_language='cn')
question2 = ts.translate_text(query_text = question1, translator = 'bing',from_language= 'cn',to_language='fr')
question3 = ts.translate_text(query_text = question2, translator = 'bing',from_language= 'fr',to_language='en')
questions_input.append(question3)
list_synonym = []
for i in range(len(questions_input)):
    list_synonym += synonym_replace_wordnet(questions_input[i])
answers_index = []
print(list_synonym)
#如果不限制次数为5的话，运行时间过长
for i in range(min(5,len(list_synonym))):
    answers_index.append(answer_tfidf(' '.join(list_synonym[i])))
    print(answers_index)
print('tfidf model进行匹配的答案如下','\n',answers[max(answers_index,key = answers_index.count)])
