import socketserver
import socket
import threading
import json
import time
import tensorflow.compat.v1 as tf
import data
import sys
import model as ml

from configs import DEFINES
from common import *

IP = "172.17.0.5"
PORT = 9005
KEY = "FIVE"
NAME = "Chatbot5"
BROKER_ADDR = ("114.70.21.90", 9000)

class Chatbot():

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            # message parsing
            message = self.rfile.readline().strip()
            message = message.decode("utf-8")
            message = json.loads(message)
            print((message['contents']))
            # message processing...
            tf.logging.set_verbosity(tf.logging.INFO)

            # 데이터를 통한 사전 구성 한다.
            char2idx,  idx2char, vocabulary_length = data.load_vocabulary()

            # 테스트용 데이터 만드는 부분이다.
            # 인코딩 부분 만든다.
            '''
            input = ""
            for i in sys.argv[1:]:
                input += i
                input += " "
            '''

            predic_input_enc, predic_input_enc_length = data.enc_processing([message['contents']], char2idx)
            predic_target_dec, _ = data.dec_target_processing([""], char2idx)

            if DEFINES.serving == True:
                # 모델이 저장된 위치를 넣어 준다.  export_dir
                predictor_fn = tf.contrib.predictor.from_saved_model(
                    export_dir=""
                )
            else:
                # 에스티메이터 구성한다.
                classifier = tf.estimator.Estimator(
                        model_fn=ml.Model, # 모델 등록한다.
                        model_dir=DEFINES.check_point_path, # 체크포인트 위치 등록한다.
                        params={ # 모델 쪽으로 파라메터 전달한다.
                            'hidden_size': DEFINES.hidden_size,  # 가중치 크기 설정한다.
                            'layer_size': DEFINES.layer_size,  # 멀티 레이어 층 개수를 설정한다.
                            'learning_rate': DEFINES.learning_rate,  # 학습율 설정한다.
                            'teacher_forcing_rate': DEFINES.teacher_forcing_rate, # 학습시 디코더 인풋 정답 지원율 설정
                            'vocabulary_length': vocabulary_length,  # 딕셔너리 크기를 설정한다.
                            'embedding_size': DEFINES.embedding_size,  # 임베딩 크기를 설정한다.
                            'embedding': DEFINES.embedding,  # 임베딩 사용 유무를 설정한다.
                            'multilayer': DEFINES.multilayer,  # 멀티 레이어 사용 유무를 설정한다.
                            'attention': DEFINES.attention, #  어텐션 지원 유무를 설정한다.
                            'teacher_forcing': DEFINES.teacher_forcing, # 학습시 디코더 인풋 정답 지원 유무 설정한다.
                            'loss_mask': DEFINES.loss_mask, # PAD에 대한 마스크를 통한 loss를 제한 한다.
                            'serving': DEFINES.serving # 모델 저장 및 serving 유무를 설정한다.
                        })

            if DEFINES.serving == True:
                predictions = predictor_fn({'input':predic_input_enc, 'output':predic_target_dec})
                #data.pred2string(predictions, idx2char)
            else:
                # 예측을 하는 부분이다.
                predictions = classifier.predict(
                    input_fn=lambda:data.eval_input_fn(predic_input_enc, predic_target_dec, DEFINES.batch_size))
                # 예측한 값을 인지 할 수 있도록
                # 텍스트로 변경하는 부분이다.
                #data.pred2string(predictions, idx2char)

            # 텍스트 문장을 보관할 배열을 선언한다.
            sentence_string = []
            # 인덱스 배열 하나를 꺼내서 v에 넘겨준다.
            if DEFINES.serving == True:
                for v in predictions['output']:
                    sentence_string = [idx2char[index] for index in v]
            else:
                for v in predictions:
                    # 딕셔너리에 있는 단어로 변경해서 배열에 담는다.
                    sentence_string = [idx2char[index] for index in v['indexs']]

            print(sentence_string)
            answer = ""
            # 패딩값도 담겨 있으므로 패딩은 모두 스페이스 처리 한다.
            for word in sentence_string:
                if word not in "<PADDING>" and word not in "<END>":
                    answer += word
                    answer += " "
            print("Chatbot5")

            # creating response message...
            self.wfile.write(answer.encode("utf-8"))


    def __init__(self):
        print("Chatbot creating...")
        main_srv = socketserver.ThreadingTCPServer(("", PORT), Chatbot.Handler, bind_and_activate=False)
        main_srv.allow_reuse_address = True
        main_srv.server_bind()
        main_srv.server_activate()
        self.srv_thread = ServerThread(main_srv)


    def start(self):
        print("Chatbot registering...")
        self.register()

        print("Chatbot running...")
        self.srv_thread.start()


    def stop(self):
        print("Chatbot deregistering...")
        self.deregister()

        print("Chatbot stopping...")
        self.srv_thread.stop()
        self.srv_thread.join()


    def register(self):
        data = { "key" : KEY, "addr" : (IP,PORT), "name" : "chatbot3"}
        sendRequest("/register?/", data, BROKER_ADDR)


    def deregister(self):
        data = { "key" : KEY }
        sendRequest("/deregister?/", data, BROKER_ADDR)


if __name__ == "__main__":
    try:
        chatbot = Chatbot()
        chatbot.start()
        while True:
            time.sleep(10)

    except KeyboardInterrupt:
        chatbot.stop()
        print("Clean up chatbot!")
