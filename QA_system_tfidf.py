import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

post_qa_question = pd.read_excel('post_qa_question.xlsx')
questions = []
answers = []
for i in range(len(post_qa_question)):
    questions.append(post_qa_question['Title'][i])
    answers.append(post_qa_question['Answer_Body'][i])

# 计算文档的TF-IDF特征
def tfidf_extractor(corpus, ngram_range=(1, 3)):
    vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True, use_idf=True, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
# 提取tfidf 特征
def conver2tfidf(data):
    new_data = []
    for q in data:
        new_data.append(q)
    tfidf_vectorizer, tfidf_X = tfidf_extractor(new_data)
    return tfidf_vectorizer, tfidf_X
# 提取tfidf 特征
tfidf_vectorizer, tfidf_X = conver2tfidf(questions)



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


def answer_tfidf(input):
    bow = tfidf_vectorizer.transform([input])
    print("bow:  ",bow)
    best_idx = idx_for_largest_cosine_sim(bow, tfidf_X)
    return answers[best_idx]

print('tfidf model进行匹配的答案如下','\n',answer_tfidf("How do I track file downloads？"))

