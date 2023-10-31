#该程序实现扩展功能
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

imp_tag = ['NN','NNS','RB','VB','VBD','VBG','VBN','JJ','JJR','JJS']
ban_word = ['do']
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
                    print("synonym_sentence[k]为：",synonym_sentence[k],tokens[i],synonym[j])
                    list = list_replace(synonym_sentence[k],tokens[i],synonym[j])
                    synonym_sentence.append(list)
    return synonym_sentence
        
questions = []
# 标题
st.header("Stackoverflow QA Extension System App")
# 输入框
question = st.text_input("Enter Question")
# post_qa_question = pd.read_excel('post_qa_question_100.xlsx')
# 点击提交按钮
if st.button("Submit"):
    #通过回译来进行扩增
    question1 = ts.translate_text(query_text = question, translator = 'bing',from_language= 'en',to_language='cn')
    question2 = ts.translate_text(query_text = question1, translator = 'bing',from_language= 'cn',to_language='fr')
    question3 = ts.translate_text(query_text = question2, translator = 'bing',from_language= 'fr',to_language='en')
    questions.append(question3)
    list_synonym = []
    for i in range(len(questions)):
        list_synonym += synonym_replace_wordnet(questions[i])
    # list_synonym = list(set(list_synonym))
    # 返回预测的值
    for i in range(len(list_synonym)):
        st.text(f" {' '.join(list_synonym[i])}")