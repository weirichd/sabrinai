from neural_net import load_data
from tensorflow.keras.models import load_model
import numpy as np
import os

def random_sentence(sentence_len=25):
    X, y, tokenizer = load_data()

    model = load_model(
        'sabrinai_2.hdf5',
        custom_objects=None,
        compile=True
    )
    arr = np.random.randint(0, len(tokenizer.word_index) + 1, size=32)
    result = []

    for i in range(sentence_len):
        next_word = model.predict_classes(arr.reshape(1, -1))
        result.append(next_word)
        arr = np.insert(arr[1:], 31, next_word, axis=0)

    sentence = tokenizer.sequences_to_texts(result)
    sentence = ' '.join(sentence).replace('\n', ' \n ')

    for c in '.!?(),':
        sentence = sentence.replace(' ' + c, c)
    return sentence


if __name__ == "__main__":
    print(random_sentence(1000))