import torch
from transformers import AlbertTokenizer, AlbertModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
ask_question = 'How do I track file downloads'
post_qa_question = pd.read_excel('post_qa_question_100.xlsx')
questions = []
answers = []
score = [] #相似度列表
for i in range(len(post_qa_question)):
    questions.append(post_qa_question['Title'][i])
    answers.append(post_qa_question['Answer_Body'][i])
    
# 加载ALBERT模型和tokenizer
model = AlbertModel.from_pretrained('albert-base-v2')
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# 将待查询问题编码为向量
encoded_ask_question = tokenizer.encode_plus(ask_question, 
                                        add_special_tokens=True, 
                                        max_length=64, 
                                        padding='max_length', 
                                        return_attention_mask=True, 
                                        return_tensors='pt')
ask_question_output = model(**encoded_ask_question)[1]
# 将问答对的问题编码为向量
for question in questions:
    encoded_question = tokenizer.encode_plus(question, 
                                            add_special_tokens=True, 
                                            max_length=64, 
                                            padding='max_length', 
                                            return_attention_mask=True, 
                                            return_tensors='pt')
    question_output = model(**encoded_question)[1]

# 计算余弦相似度
sim_score = torch.nn.functional.cosine_similarity(ask_question_output, question_output, dim=1)
score.append(sim_score)
print('最匹配的答案是',answers[score.index(max(score))])

