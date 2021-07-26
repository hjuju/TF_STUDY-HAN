from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from icecream import ic

docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요', '추천하고 싶은 영화입니다',
        '한 번 더 보고 싶네요', '글세요', '별로에요', ' 생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '청순이가 잘 생기긴 했어요']

# 문장의 개수 = 행의 개수

# 긍정 1, 부정 0 
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
ic(token.word_index)

x = token.texts_to_sequences(docs)
ic(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences # 데이터가 크기가 다른 데이터셋에 작은것에 0을 넣어줌
from tensorflow.keras.utils import to_categorical

pad_x = pad_sequences(x, padding='pre', maxlen=5) # pre: x 데이터에 대해서 앞부분에 0을 채워줌 / 가장 큰 데이터수 보다 적은 maxlen을 주면 앞 부분부터 잘림
ic(pad_x)
ic(pad_x.shape) # (13, 5) 

word_size = len(token.word_index) 
ic(word_size) # 27(단어의 종류가 27개)
ic(len(np.unique(pad_x)))

# 원 핫 인코딩 -> 라벨의 개수만큼 생김 (13, 5) -> (13, 5, 27)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,LSTM, Embedding, Input #  -> onehot, vector화 까지 Embedding이 모두 처리해줌

# model = Sequential()
# model.add(Embedding(input_dim=28, output_dim=77, input_length=5)) # inputdim * ouputdim // input_length는 파라미터 개수에 영향 X
'''
# # input_dim -> 라벨(단어사전)의 개수(자연어처리 -> 단어사전의 개수(word_size)) / output_dim(인베딩 레이어만 아웃풋위치가 가운데) / input_length -> (max_len)단어수, 문장 길이(x.shape의 열 부분) 
# # model.add(Embedding(28, 77)) # input, output / input_lengh는 명시하지 않아도 자동으로 인식해줌
# # model.add(Embedding(28, 77, input_length=5))
# # input_dim 과 input_length의 값은 크게 줘도 상관없지만 파라미터 수 증가, 각각 권장된 대로 넣어주는게 좋음
'''
# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))

input = Input(shape=(5,))
Em1 = Embedding(input_dim=28, output_dim=77, input_length=5)(input)
x2 = LSTM(32)(Em1)
output = Dense(1, activation='sigmoid')(x2)

model = Model(inputs=input, outputs=output)

model.summary()

'''
Sequence
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 77)             2079
_________________________________________________________________
lstm (LSTM)                  (None, 32)                14080
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 16,192
Trainable params: 16,192
Non-trainable params: 0
_________________________________________________________________
'''

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=1)

#4. 평가, 예측

acc = model.evaluate(pad_x, labels)[1]
ic(acc)