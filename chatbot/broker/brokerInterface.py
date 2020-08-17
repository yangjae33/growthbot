# -*- coding: utf-8 -*-

import socketserver
import socket
import random
import json
import os
import sys
import base64
import shutil
from usedfunc import *
from time import sleep
from time import time

#broker PORT
PORT = 9000


class Broker():
    #dictionary for chatbots (chatbot info)
    chatbots = ChatbotDict()

    class Handler(socketserver.StreamRequestHandler):
        #message routing rules
        funcMap = {
            "/register" : registerChatbot,  #register chatbot to broker
            "/deregister" : deregisterChatbot, #delete chatbot from broker
            "/message" : dispatchMessage,       #sending message to broker
            "/get_chatbot_info": getChatbotInfo #get registered chatbot info
        }


        def handle(self):
            #receive message
            message = self.rfile.readline().strip()
            message = message.decode("utf-8")

            #split message by '?', cmd - command, data - message data
            #request 메세지 : "/message?/"+ "contents" + "video_name"
            cmd, data = message.split("?/")

            #write message data to json file
            data = json.loads(data)

            #check if command is in Function Map
            if cmd in Broker.Handler.funcMap:
                response = Broker.Handler.funcMap[cmd](Broker.chatbots, data)
            else:
                response = {"contents" : None}

            #write response to json file
            response = json.dumps(response)
            self.wfile.write(response.encode("utf-8"))


    def __init__(self):
        print("broker creating...")

        #create server
        main_srv = socketserver.ThreadingTCPServer(("", PORT), Broker.Handler, bind_and_activate=False)
        main_srv.allow_reuse_address = True
        main_srv.server_bind()
        main_srv.server_activate()
        self.srv_thread = ServerThread(main_srv)
        shutil.rmtree("./data_out")
        os.mkdir("./data_out")
       
        #usedfunctions.py
        training()


    #start broker
    def start(self):
        print("broker running...")
        self.srv_thread.start()


    #stop broker
    def stop(self):
        print("broker stopping...")
        self.srv_thread.stop()
        self.srv_thread.join()


if __name__ == "__main__":
    try:
        #create Broker() object
        broker = Broker()

        #start broker
        broker.start()

        while True:
            sleep(1)

    except KeyboardInterrupt:
        broker.stop()
        print("Clean up broker!")


