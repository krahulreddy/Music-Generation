import os
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding
#from keras.utils.vis_utils import plot_model

cij = "cti.json"
model_weights_directory = 'Model_Weights1/'


def generateSequence(initial_index, seq_length):
    with open(cij) as f:
        char_to_index = json.load(f)
    itc = {i:ch for ch, i in char_to_index.items()}
    allChars = len(itc)
    model = buildModel(allChars)
    print(model.summary())
    model.load_weights(model_weights_directory + "Weights_90.h5")
    sequence_index = [initial_index]
    for i in range(seq_length):
        batch = np.zeros((1, 1))
        batch[0, 0] = sequence_index[-1]
        prediction = model.predict_on_batch(batch).ravel()
#        print(prediction)
        sample = np.random.choice(range(allChars), size = 1, p = prediction)
#        print(sample, itc[np.argmax(prediction)])
        sequence_index.append(sample[0])
    seq = ''
    for c in sequence_index:
        seq = seq + itc[c]
    cnt = 0
    for i in seq:
        cnt += 1
        if i == "\n":
            break
    seq1 = seq[cnt:]
    cnt = 0
    for i in seq1:
        cnt += 1
        if i == "\n" and seq1[cnt] == "\n":
            break
    seq2 = seq1[:cnt]
    return seq2

def buildModel(unique_chars):
    model = Sequential()
    model.add(Embedding(input_dim = unique_chars, output_dim = 512, batch_input_shape = (1, 1)))
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences = True, stateful = True,))
    model.add(Dropout(0.2))
    model.add(LSTM(256, stateful = True)) 
    model.add(Dropout(0.2))    
    model.add((Dense(unique_chars)))
    model.add(Activation("softmax"))
#    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model

if __name__ == '__main__':
    sequence = generateSequence(8, 550)
    print("Music Generated : \n")
    print(sequence)