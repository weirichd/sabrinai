#
# Run this after running get_scripts.py
#

#
# Run this after running get_scripts.py
#

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Activation, Softmax

import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback


def clean_text(t):
    t = t.strip()
    for c in '.!?(),':
        t = t.replace(c, ' ' + c)
    t = t.replace('\n', ' \n ')

    return t


episodes_path = 'episodes'
sequence_length = 32

texts = [clean_text(open(os.path.join(episodes_path, f)).read()) for f in os.listdir(episodes_path)]

tkn = Tokenizer(num_words=50000, filters='"#$%&*+-/:;<=>@[\\]^_`{|}~\t')

tkn.fit_on_texts(texts)

seqs = tkn.texts_to_sequences(texts)

vocab_size = len(tkn.word_index) + 1

short_sequences = []
for seq in seqs:
    short_sequences += [(seq[i:i+sequence_length], seq[i+sequence_length])
                        for i in range(len(seq) - sequence_length )]

X = [a[0] for a in short_sequences]
y = [a[1] for a in short_sequences]

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=32))
model.add(LSTM(128))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(vocab_size))
model.add(Softmax())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

X = np.array(X)
y = np.array(y)


def random_sentence(sentence_len=25):
    arr = np.random.randint(0, vocab_size, size=32)
    result = []

    for i in range(sentence_len):
        next_word = model.predict_classes(arr.reshape(1, -1))
        result.append(next_word)
        arr = np.insert(arr[1:], 31, next_word, axis=0)

    sentence = tkn.sequences_to_texts(result)
    sentence = ' '.join(sentence)

    for c in '.!?(),':
        sentence = sentence.replace(' ' + c, c)

    print()
    print(sentence)
    print()


callbacks=[LambdaCallback(on_epoch_end=lambda epoch, logs: random_sentence(20))]


model.fit(X[:X.shape[0]], y[:X.shape[0]],
          epochs=10,
          batch_size=100,
          callbacks=callbacks)

model.save('sabrinai.hdf5')

random_sentence(1000)
