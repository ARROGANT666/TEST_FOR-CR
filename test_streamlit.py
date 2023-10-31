import streamlit as st
import pandas as pd
import joblib
from sklearn.naive_bayes import GaussianNB
post_qa_question = pd.read_excel('d:\.毕业论文\easy_py\post_qa_question.xlsx')

# 标题
st.header("Streamlit Machine Learning App")

df = pd.DataFrame({"Height":['88.9', '90.2', '82.7', '81.4', '83.5'],
"Weight":['48.3', '47.4', '44.8', '48.2', '39.9'],
"Species":['Dog', 'Dog', 'Dog', 'Dog', 'Dog'],
},
columns =['Height','Weight','Species'])
# 输入框
height = st.number_input("Enter Height")
weight = st.number_input("Enter Weight")
X = df[["Height", "Weight"]]
y = df["Species"]

clf = GaussianNB() 
clf.fit(X, y)
joblib.dump(clf, "clf.pkl")
# 点击提交按钮
if st.button("Submit"):
    # 引入训练好的模型
    clf = joblib.load("clf.pkl")
 
    # 转换成DataFrame格式的数据
    X = pd.DataFrame([[height, weight]],
                    columns=["Height", "Weight"])
 
    # 获取预测出来的值
    prediction = clf.predict(X)[0]
 
    # 返回预测的值
    st.text(f"This instance is a {prediction}")