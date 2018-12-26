import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Embedding,Bidirectional,Dropout,Dense,GlobalMaxPool1D
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd


#import the training dataset
train = pd.read_csv("train.csv") 

comm_train = train["comment_text"]
y_train = train[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].values

#convert texts to number sequence using tokenizer api
token = Tokenizer(num_words=50000)
token.fit_on_texts(comm_train)
inp = token.texts_to_sequences(comm_train)

#Pad each comment
max_len=100
x_train = pad_sequences(inp, max_len)

words = token.word_index.keys()
vocab_size = len(words) + 1

model = Sequential()
#Add embedding layer for word embedding
e = Embedding(vocab_size, 300, input_length=max_len, trainable=True)
model.add(e)
model.add(Bidirectional(LSTM(256, activation='relu',return_sequences=True)))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=5)



