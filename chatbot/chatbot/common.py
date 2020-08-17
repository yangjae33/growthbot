import base64
import os
import threading
import time
import json
import random
import socket

def registerChatbot(chatbots, data):
    chatbots.lock.acquire()

    key = data["key"]
    del data["key"]
    chatbots[key] = ChatbotInfo(data)

    chatbots.lock.release()
    return {"contents" : "chatbot registered"}

def deregisterChatbot(chatbots, data):
    chatbots.lock.acquire()
    key = data["key"]
    if key in chatbots.keys():
        del chatbots[key]

    chatbots.lock.release()
    return {"contents" : "chatbot deregistered"}

def dispatchMessage(chatbots, data):

    # Select Chatbot
    allChat = list(chatbots.keys())
    dstChatbot = random.choice(allChat)
    address = chatbots[dstChatbot].address
    received = sendRequest("", data, address)
    return received

def urandom():
    random_bytes = os.urandom(16)
    token = base64.b16encode(random_bytes).decode('utf-8')
    return token

def sendRequest(method, data, address):
    requestMsg = method + json.dumps(data) + "\n"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(address)
            s.sendall(requestMsg.encode("utf-8"))
            received = str(s.recv(1024), "utf-8")
            
    except socket.error as e:
        print(e)
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
        self.ts = time.time()
        self.address = tuple(attrs["addr"])
        self.attrs = attrs
        
    def __str__(self):
        return "{0}, {1}".format(self.ts, self.attrs)

    
class ChatbotDict(dict):
    def __init__(self):
        self.lock = threading.Lock()
