import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq
import io
import glob

print("start")

#path = 'AjayKumar.docx'
text = ""
files = [file for file in glob.glob(r"C:\Users\SANJAY AGARWAL\Desktop\cv\*")]
for file_name in files:
    with io.open(file_name, encoding='cp437', errors='ignore', newline='') as image_file:
        text = image_file.read().lower() + text
print('corpus length:', len(text))

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

WORD_LENGTH = 5
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])
#print(prev_words[0])
#print(next_words[0])

X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1



#print(X[0][0])


from numpy import asarray
from numpy import save
data = asarray(X)

save('features.npy', data)
print(" features done")


data = asarray(Y)
save('label.npy', data)
print("label done")
