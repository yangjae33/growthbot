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
import nltk
import threading
import sys
import gensim
import Word2Vec

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
    #open file with information of registered chatbots
    with open('data_in/chatbot_info_file.json', 'r') as f:
        #data of registered videos at its chatbots
        list_of_data = json.load(f)

        #list of videos registered in MOOCACHA which have registered chatbot
        list_of_videos = list(list_of_data.keys())

    #find all videos registered in the broker
    for video in list_of_videos:
        #data file for training
        train_data_file = 'data_in/'+video+'_train_data.txt'

        #trained model file
        trained_model = 'data_out/'+video+'_model.pkl'

        #if trained model of video is saved
        if os.path.isfile(trained_model):
            print(video, "-->Model is trained!")

        #if trained model of video is not saved
        else:
            print("Start training...-->", video)

            #open training data file
            train_data = W2V.read_data(train_data_file)

            #open testing data file
            test_data = W2V.read_data("data_in/broker_test_dataset.txt")

            #train data frame from train data (ChatNum - chatbot class number, Question - questions)
            df_Chatbot = pd.DataFrame({
                "ChatNum" : [ train_data[i][1] for i in range(len(train_data))],
	        "Question" : [ train_data[i][0] for i in range(len(train_data))],
	    })

            #list of chatbot classes
            Chatbot_class = [
        	"1" if df_Chatbot.iloc[i]['ChatNum'] == '1'
	        else
        	"2" if df_Chatbot.iloc[i]['ChatNum'] == '2'
	        else
	        "3"
        	for i in range(df_Chatbot.shape[0])
	    ]

            #write list of classes to train data frame
            df_Chatbot["class"] = Chatbot_class

            #write train data frame to pickle
            df_Chatbot.to_pickle("data_out/"+video+"_df_Chatbot_chat.pkl")   #changed

            #read train data frame pickle
            df_Chatbot = pd.read_pickle("data_out/"+video+"_df_Chatbot_chat.pkl")

            #test data frame
            df_test = pd.DataFrame({
        	"X_test" : [ test_data[i][0] for i in range(len(test_data))],
	        "y_test" : [ test_data[i][1] for i in range(len(test_data))],
	    })

            #tokenize chatbot questions and write to train data frame
            df_Chatbot['token_chat'] = df_Chatbot['Question'].apply(tokenizer_twitter_morphs)

            #tokenize schatbot questions from test dataset and write to test data frame
            df_test['X_test_tokkened'] = df_test['X_test'].apply(tokenizer_twitter_morphs)

            #return first row of train data frame
            df_Chatbot.head()

            #tokens from train data frame
            tokens = [ t for d in df_Chatbot['token_chat'] for t in d]

            #NLP of tokens
            text = nltk.Text(tokens, name='NMSC')

            #takes a group of rows and columns by labels
            X_train = df_Chatbot.loc[:, 'Question'].values
            y_train = df_Chatbot.loc[:, 'ChatNum'].values

            #convert words to numbers
            tfidf = TfidfVectorizer(tokenizer=tokenizer_twitter_morphs)

            #make pipeline
            multi_nbc = Pipeline([('vect', tfidf), ('nbc', MultinomialNB())])

            #time: start training
            start = time()

            #Naive Bayes Classification
            multi_nbc.fit(X_train, y_train)

            #time: finish training
            end = time()
            print('Time: {:f}s'.format(end-start))

            #evaluate, test
            y_pred = multi_nbc.predict(df_test["X_test"])
            print("테스트 정확도: {:.3f}".format(accuracy_score(df_test["y_test"], y_pred)))

            #model of Naive Bayes Classification
            model = make_pipeline(TfidfVectorizer(), MultinomialNB())
            model.fit(X_train, y_train)
            labels = model.predict(df_test["X_test"])

            #save model to pickle file
            filename = 'data_out/'+video+'_model.pkl'
            pickle.dump(model, open(filename, 'wb'))
            print('Training is finished!')
    return 'Training is finished'


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
    #title of current video
    video_name = data["video_name"]

    #open trained model file
    train_file = open('data_out/'+video_name+'_model.pkl', 'rb')
    model = pickle.load(train_file)

    #open train data file
    file = open('data_out/'+video_name+'_df_Chatbot_chat.pkl', 'rb')
    train = pickle.load(file)

    #open registered chatbot information file
    chatbot_info_file = open('data_in/chatbot_info_file.json', 'r')
    chatbot_info = json.load(chatbot_info_file)

    #received message
    request = data["contents"]

    #predict chatbot class
    predict = predict_category(request, train, model)

    #predicted chatbot's adress
    address = chatbot_info[video_name][predict]["chatbot_ip"]
    PORT = chatbot_info[video_name][predict]["chatbot_port"]
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
