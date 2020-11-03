
from keras.models import load_model
import numpy as np
from auto_data import prepare_input
from auto_data import sample
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
model = load_model('keras_next_word_model.h5')

print('start')
from numpy import load
unique_words = load('unique_words.npy')

print('wohoh')

def predict_completions(text, n=3):
    if text == "":
        return("0")
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [unique_words[idx] for idx in next_indices]
q =  "I"
print("correct sentence: ",q)
seq = " ".join(tokenizer.tokenize(q.lower())[0:5])
print("Sequence: ",seq)
print("next possible words: ", predict_completions(seq, 5))
