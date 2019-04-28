# %%
import tensorflow as tf

tf.enable_eager_execution()

import numpy as np

import os
import time

# step 1: prepare data

# %%
# download file
path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# %%
text = open(path_to_file).read()
print('Length of text: {} characters'.format(len(text)))

# %%
print(text[:100])

# %%
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# %% create char array and int(represent char) array
char2idx = {u: i for i, u in enumerate(vocab)}

idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

# %%
# note: zip(list, range(n))
for char, _ in zip(char2idx, range(20)):
    print('{:6s} ---> {:4d}'.format(repr(char), char2idx[char]))
print('{} ---- characters mapped to int ---- > {}'.format(text[:13], text_as_int[:13]))

# step 2: train
# %%
# suppose text = "hello"
# train sample: hell, target sample: ello
seq_length = 10
chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length + 1, drop_remainder=True)

# %%
# note: repr()
for item in chunks.take(2):
    print(repr(''.join(idx2char[item.numpy()])))


# %%
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = chunks.map(split_input_target)


