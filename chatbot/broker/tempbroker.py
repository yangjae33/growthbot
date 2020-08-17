# -*- coding: utf-8 -*- 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB # 다항분포 나이브 베이즈 모델
from sklearn.metrics import accuracy_score #정확도 계산
import pandas as pd
import numpy as np

from sklearn import preprocessing

data_df = pd.read_csv("./data2.csv", header=0)

answer,label = data_df['A'], data_df['label']

dtmvector = CountVectorizer()
X_train_dtm = dtmvector.fit_transform(answer)
print(X_train_dtm.shape)
tfidf_transformer = TfidfTransformer()
tfidfv = tfidf_transformer.fit_transform(X_train_dtm)
#print(X_train_dtm.shape)

mod = MultinomialNB()
mod.fit(tfidfv,label)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

#newsdata_test = fetch_20newsgroups(subset='test', shuffle=True) #테스트 데이터 갖고오기
X_test_dtm = dtmvector.transform(answer) #테스트 데이터를 DTM으로 변환
tfidfv_test = tfidf_transformer.transform(X_test_dtm) #DTM을 TF-IDF 행렬로 변환

predicted = mod.predict(tfidfv_test) #테스트 데이터에 대한 예측
#print(tfidfv_test)
print("정확도:", accuracy_score(label, predicted)) #예측값과 실제값 비교

# training completed
#==========================================================

#print(testst)
print()
df = pd.read_csv("./tokenized_text.csv",header = 0)
testst = df['W'][-20:]
#print(testst)
X_test_dtm = dtmvector.transform(testst)
tfidfv_test = tfidf_transformer.transform(X_test_dtm) #DTM을 TF-IDF 행렬로 변환

predicted = mod.predict(tfidfv_test) #테스트 데이터에 대한 예측
print(predicted)
print("예측값:",sum(predicted)/testst.size)

#===========================================================

total_video_len = 45+33*60
curr_video_len = 41+3*60

df = pd.read_csv("./tokenized_text.csv",header = 0)
df = df['W']
#bias = np.round((df.size-1)/total_video_len*1)
bias = 10
t = np.floor(curr_video_len/total_video_len*(df.size-1))
wt = df[t]
#print(wt)
leftend = t-bias
rightend = t+bias
if rightend>df.size-1:
    rightend = df.size-1
if leftend <1:
    leftend = 1
    
#print(df[int(leftend):int(rightend)])

pred = df[int(leftend):int(rightend)]
X_pred_dtm = dtmvector.transform(pred)
tfidfv_pred = tfidf_transformer.transform(X_pred_dtm)
predicted = mod.predict(tfidfv_pred) 
#print(predicted)
ques = "추상화가 뭐야?"
#print("질문 내용 : 추상화가 뭐야?")
print("질문 시간 : ",curr_video_len,"초")
print("예측값:","챗봇(",int(round(sum(predicted)/pred.size)),")")
