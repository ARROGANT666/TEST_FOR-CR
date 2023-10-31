#该程序同时实现问题扩展和答案抽取两个功能，使用streamlit来搭建网页
'''
pip install streamlit
然后在Anaconda prompt输入相应的streamlit指令
'''
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.text import Text
from nltk import pos_tag
from nltk.corpus import stopwords
from multiprocessing import Pool
from tqdm import *
from nltk.corpus import wordnet as wn
import numpy as np
from scipy.stats import entropy
import math
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)
# 需要用绝对路径，如果用相对路径会报错，可能是用了streamlit框架的原因
post_qa_question = pd.read_excel('d:\.毕业论文\qa_system\post_qa_question_100.xlsx')

questions = []
answers = []
for i in range(len(post_qa_question)):
    questions.append(post_qa_question['Title'][i])
    answers.append(post_qa_question['Answer_Body'][i])


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
# print(tfidf_X)
# print('tfidf_vectorizer:',tfidf_vectorizer,'tfidf_X:',tfidf_X)
# print('TFIDF model')
# print('vectorizer',tfidf_vectorizer.get_feature_names())
# print('vector of text',tfidf_X[0:3].toarray())


        
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
    # print("bow:  ",bow)
    best_idx_cos = idx_for_largest_cosine_sim(bow, tfidf_X)
    return best_idx_cos

#重述核心代码
def paraphrase(
    question,
    num_beams=10,
    num_beam_groups=10,
    num_return_sequences=10,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    input_ids = input_ids.to(device)  # 将输入移动到GPU上

    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res


# 标题
st.header("Stackoverflow QA  System App")
# 输入框
question_input = st.text_input("Enter Question")

# 点击提交按钮
if st.button("使用问题扩展进行答案生成"):
    list_synonym = paraphrase(question_input)
    # 返回预测的值
    st.write("扩展后问题如下：")
    for i in range(len(list_synonym)):
        st.text(f" {list_synonym[i]}")
    st.write("问题答案如下：")
    answers_index = []
    #如果不限制次数为5的话，运行时间过长
    for i in range(min(5,len(list_synonym))):
        answers_index.append(answer_tfidf(list_synonym[i]))
    st.text(f"{answers[max(answers_index,key = answers_index.count)]}")
if st.button("直接进行答案生成"):
    st.write("问题答案如下：")
    answers_index = answer_tfidf(question_input)
    st.text(f"{answers[answers_index]}")

