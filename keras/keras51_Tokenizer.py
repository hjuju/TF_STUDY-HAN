from tensorflow.keras.preprocessing.text import Tokenizer # 자연어(문자)를 수치화 시킴
from icecream import ic


text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹었다.'
#        3    1    4    5     6   1    2    2    7     3145 6 / 1456 1 / 4562 2 / .. / 6122 7 나눌 수 있음 -> LSTM, DNN, ConV1,2D

token = Tokenizer()
token.fit_on_texts([text]) # 가져 온 데이터를 토큰화 

ic(token.word_index)
# {'나는': 3, '마구': 2, '맛있는': 5, '매우': 4, '먹었다': 7, '밥을': 6, '진짜': 1}

x = token.texts_to_sequences([text]) 
ic(x) # [[3, 1, 4, 5, 6, 1, 2, 2, 7]] 수치화 된 형태로 출력

# 원 핫 인코딩 
from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
ic(word_size) # 7개

x = to_categorical(x) # 카테고리컬 이기 때문에 0부터 채워짐
ic(x, x.shape) # shape가 1,9,8인 이유 -> 위에 제시 된 텍스트가 9개 이기 때문에 9개의 행 발생
