import tensorflow as td
from tensorflow import keras
import numpy as np

import Constants

data = keras.datasets.imdb

# Get the top 10,000 frequent words. Shrinks our data.
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=Constants.SizeOfWordRank)

# The dataset compresses the data with dictionary compression.
# This line of code retrieves the dictionary.
word_index = data.get_word_index()

# The dictionary has some additional elements we need to add.
# Let's call these "control codes".
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNKNOWN>"] = 2  # When something isn't found
word_index["<UNUSED>"] = 3

# Reverse the dictionary. Right now the key is the string, and the value is the integer representation
# This makes the integer the key and the string the value.
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# Decodes the array into a string.
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# print(decode_review(test_data[0]))

# Make the length of all the data set to 250.
# If it's more than 250, truncate the data. If it's less, pad the data.
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=Constants.SizeOfReviews)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=Constants.SizeOfReviews)


'''
Here we set up the neurological model.
1 - Input layer. This is where we input our review. The review is a fixed length of 250.
2 - 
3 - 
4 - Boolean output. Determine whether the review is positive or negative.

'''
model = keras.Sequential()
model.add(keras.layers.Embedding(Constants.SizeOfWordRank, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model
