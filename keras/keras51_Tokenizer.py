from tensorflow.keras.preprocessing.text import Tokenizer # 자연어(문자)를 수치화 시킴
from icecream import ic


text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'
#        3    1    4    5     6   1    2    2    7     3145 6 / 1456 1 / 4562 2 / .. / 6122 7 나눌 수 있음 -> LSTM, DNN, ConV1,2D

token = Tokenizer()
token.fit_on_texts([text]) # 가져 온 데이터를 토큰화 

ic(token.word_index)