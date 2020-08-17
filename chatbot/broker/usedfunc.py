# -*- encoding: utf-8 -*-

import base64
import os
import threading
import json
import random
import socket
import pandas as pd
import numpy as np
import pickle
import threading
import sys
import gensim
import Word2Vec

# -*- coding: utf-8 -*- 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from konlpy.tag import Okt
from time import time


twitter = Okt() #KONLPY tokenizer
W2V = Word2Vec.Word2Vec()

#functions for training
def tokenizer_twitter_morphs(doc):
    return twitter.morphs(doc)

def tokenizer_twitter_noun(doc):
    return twitter.nouns(doc)

def tokenizer_twitter_pos(doc):
    return twitter.pos(doc, norm=True, stem=True)

def predict_category(s, train, model):
    pred = model.predict([s])
    return pred[0]

#train data from chatbots
def training():
    data_df = pd.read_csv("./data2.csv", header=0)

    answer,label = data_df['A'], data_df['label']

    dtmvector = CountVectorizer()
    X_train_dtm = dtmvector.fit_transform(answer)
    print(X_train_dtm.shape)
    tfidf_transformer = TfidfTransformer()
    tfidfv = tfidf_transformer.fit_transform(X_train_dtm)
    
    mod = MultinomialNB()
    mod.fit(tfidfv,label)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

    #테스트 데이터를 DTM으로 변환
    X_test_dtm = dtmvector.transform(answer) 
    #DTM을 TF-IDF 행렬로 변환
    tfidfv_test = tfidf_transformer.transform(X_test_dtm) 

    #테스트 데이터에 대한 예측
    predicted = mod.predict(tfidfv_test)
    print("Accuracy :", accuracy_score(label, predicted)) #예측값과 실제값 비교

    return 'Training is finished'

# training completed
#==========================================================


#register chatbot
def registerChatbot(chatbots, data):
    chatbots.lock.acquire()
    key = data["key"]
    del data["key"]
    chatbots[key] = ChatbotInfo(data)
    chatbots.lock.release()
    print({"contents" : "chatbot registered"})
    return {"contents" : "chatbot registered"}


#deregister chatbot
def deregisterChatbot(chatbots, data):
    chatbots.lock.acquire()
    key = data["key"]
    if key in chatbots.keys():
        del chatbots[key]
    chatbots.lock.release()
    print({"contents" : "chatbot deregistered"})
    return {"contents" : "chatbot deregistered"}

#dispatch message
def dispatchMessage(chatbots, data):
    data_df = pd.read_csv("./data2.csv", header=0)

    answer,label = data_df['A'], data_df['label']

    dtmvector = CountVectorizer()
    X_train_dtm = dtmvector.fit_transform(answer)
    print(X_train_dtm.shape)
    tfidf_transformer = TfidfTransformer()
    tfidfv = tfidf_transformer.fit_transform(X_train_dtm)
    
    # training (temp)
    #================

    total_video_len = data['total_time']
    curr_video_len = data['time']

    df = pd.read_csv("./tokenized_text.csv",header = 0)
    df = df['W']
    #bias = np.round((df.size-1)/total_video_len*1)
    bias = 10
    t = np.floor(curr_video_len/total_video_len*(df.size-1))

    leftend = t-bias
    rightend = t+bias
    if rightend>df.size-1:
        rightend = df.size-1
    if leftend <1:
        leftend = 1



    mod = MultinomialNB()
    mod.fit(tfidfv,label)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

    #테스트 데이터를 DTM으로 변환
    X_test_dtm = dtmvector.transform(answer) 
    #DTM을 TF-IDF 행렬로 변환
    tfidfv_test = tfidf_transformer.transform(X_test_dtm) 
    
    #테스트 데이터에 대한 예측
    predicted = mod.predict(tfidfv_test)
    print("Accuracy :", accuracy_score(label, predicted)) #예측값과 실제값 비교


    pred = df[int(leftend):int(rightend)]
    print(pred)
    X_pred_dtm = dtmvector.transform(pred)
    tfidfv_pred = tfidf_transformer.transform(X_pred_dtm)
    predicted = mod.predict(tfidfv_pred) 
    pred_chatbot_num = int(round(sum(predicted)/pred.size))
    #print(predicted)
    ques = "추상화가 뭐야?"
    #print("질문 내용 : 추상화가 뭐야?")
    print("질문 시간 : ",curr_video_len,"초")
    print("예측값:","챗봇(",pred_chatbot_num,")")

    # #title of current video
    # video_name = data["video_name"]

    # #open trained model file
    # train_file = open('data_out/'+video_name+'_model.pkl', 'rb')
    # model = pickle.load(train_file)

    # #open train data file
    # file = open('data_out/'+video_name+'_df_Chatbot_chat.pkl', 'rb')
    # train = pickle.load(file)

    #open registered chatbot information file
    chatbot_info_file = open('data_in/chatbot_info.json', 'r')
    chatbot_info = json.load(chatbot_info_file)

    # #predict chatbot class
    # predict = predict_category(request, train, model)

    #predicted chatbot's adress
    address = chatbot_info[str(pred_chatbot_num)]["chatbot_ip"]
    PORT = chatbot_info[str(pred_chatbot_num)]["chatbot_port"]
    print("connects with: ", address, str(PORT))

    #receiving answer from chatbot
    received = sendRequest("", data, address, PORT)
    return received


#get information of registered chatbot
def getChatbotInfo(chatbots, data):
    #video name for which chatbot is purposed
    video_name = data["contents"]["video_name"]

    #name of chatbot which is beeing added
    chatbot_name = data["contents"]["chatbot_name"]

    #number of chatbot class; used for predicting chatbot
    class_number = 1

    #training data file
    file = video_name + "_train_data.txt"

    #if training data exists
    if os.path.isfile("data_in/"+file):
        #open training data file
        train_file = open("data_in/" + file, 'r')

        #find first line of file
        first_line = train_file.readlines(1)

        #if file is empty
        if not first_line:
            #open file for writing new data
            train_file = open("data_in/" + file, 'w')

            #write train data of chatbot line by line with class number
            for item in data["contents"]["chatbot_info"]["train_data"]:
                train_file.writelines(item.replace('\n','') + "\t" + str(class_number) + "\n")
                chatbot_status = str(class_number)

        #if file has data
        else:
            #read last line
            last_line = train_file.readlines()[-1]

            #split line by '\t' to find last chatbot's class number
            split_line = last_line.split('\t')
            class_num = split_line[-1].replace('\n','')

            #count new chatbot's class
            class_number = int(class_num) + 1
            chatbot_status = str(class_number)

            #open training data file for apending new data line by line with class number
            train_file = open("data_in/"+file, 'a')
            for item in data["contents"]["chatbot_info"]["train_data"]:
                train_file.writelines(item.replace('\n','') + "\t" + str(class_number) + "\n")

    #if training data file does not exist
    else:
        #open training data file for writing new data line by line with class number
        train_file = open("data_in/"+file, 'w')
        for item in data["contents"]["chatbot_info"]["train_data"]:
            train_file.writelines(item.replace('\n','') + "\t" + str(class_number) + "\n")
            chatbot_status = str(class_number)
            print("chatbot class: ", chatbot_status)

    #json for new registered chatbot information
    chatbot_class = {chatbot_status: data["contents"]["chatbot_info"]}
    chatbot_class[chatbot_status]["chatbot_name"] = chatbot_name

    #read file of chatbot information file
    with open('data_in/chatbot_info_file.json', 'r', encoding='utf8') as f:
        old_info = json.load(f)

    #if current video is in chatbot information file, append new data
    if video_name in old_info:
        old_info[video_name].update(chatbot_class)

    #if current video is not in chatbot information file, add new data
    else:
        old_info[video_name] = chatbot_class

    #open chatbot information file for writing, write chatbot inforamtion
    with open('data_in/chatbot_info_file.json', 'w') as output:
        json.dump(old_info, output)

    #trained model file
    trained_model = 'data_out/'+video_name+'_model.pkl'

    #print if trained model is saved or not
    if os.path.isfile(trained_model):
        pass
    else:
        print("Training is required!")


#send reques to predicted chatbot
def sendRequest(method, data, address, PORT):
    requestMsg = method + json.dumps(data) + "\n"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((address, PORT))
            s.sendall(requestMsg.encode("utf-8"))
            received = str(s.recv(1024), "utf-8")
    except socket.error as e:
        print("챗봇이 존재하지 않거나 꺼져있다")
        received = -1
    return received


class ServerThread(threading.Thread):
    def __init__(self, server):
        super(ServerThread, self).__init__()
        self.server = server

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class ChatbotInfo():
    def __init__(self, attrs):
        self.ts = time()
        self.address = tuple(attrs["addr"])
        self.attrs = attrs


    def __str__(self):
        return "{0}, {1}".format(self.ts, self.attrs)


class ChatbotDict(dict):
    def __init__(self):
        self.lock = threading.Lock()
