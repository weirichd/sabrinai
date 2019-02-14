#############################################
#
# Run this after running get_episodes.py
#
#############################################

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Activation, Softmax

import os


def clean_episode_text(text, header_length=0):
    """
    Clean the text of an episode.

    :param text:
    :param header_length:
    :return:
    """

    # Remove any leading newlines or whitespaces at the beginning and end of the episode text.
    text = text.strip()

    # Insert a space before these punctuation marks so they get treated as a word.
    for c in '.!?),':
        text = text.replace(c, ' ' + c)

    # Treat these two a special cases
    text = text.replace(')', ' )')
    text = text.replace('\n', ' \n ')

    # If the "cleaning" I just did put two or more spaces next to each other, get rid of that.
    while '  ' in text:
        text = text.replace('  ', ' ')

    text = ('episodebeginning ' * header_length) + text + (' episodeend' * header_length)

    return text


def load_data(episodes_path='episodes', sequence_length=32):
    texts = [clean_episode_text(open(os.path.join(episodes_path, f)).read()) for f in os.listdir(episodes_path)]

    tkn = Tokenizer(num_words=50000, filters='"#$%&*+-/:;<=>@[\\]^_`{|}~\t')

    tkn.fit_on_texts(texts)

    seqs = tkn.texts_to_sequences(texts)

    sub_texts = []
    for seq in seqs:
        sub_texts += [
            (seq[i:i+sequence_length], seq[i+sequence_length], [j/len(seq) for j in range(i, i+sequence_length)])
            for i in range(len(seq) - sequence_length)
        ]

    X = np.array([s[0] for s in sub_texts])
    y = np.array([s[1] for s in sub_texts])

    return X, y, tkn


X, y, tokenizer = load_data()
vocab_size = len(tokenizer.word_index) + 1


def build_mode(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=32))
    model.add(LSTM(128))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(vocab_size))
    model.add(Softmax())

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    return model


model = build_mode(vocab_size)


def random_sentence(tkn, sentence_len=25):
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

    sentence = sentence.replace('\n', ' \n ')

    return sentence


callbacks = [LambdaCallback(on_epoch_end=lambda epoch, logs: print('\n', random_sentence(tokenizer, 50), '\n'))]

model.fit(X[:100000], y[:100000],
          epochs=32,
          batch_size=100,
          callbacks=callbacks)

model.save('sabrinai_2.hdf5')

print(random_sentence(tokenizer, 1000))
