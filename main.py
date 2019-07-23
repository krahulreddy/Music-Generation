import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding

batchSize=16
seqLength=64

def makeBatch(allChars,uniqueChars):
    size = allChars.shape[0]
    batchCapacity = int(size/ batchSize)
    
    for s in range(0, batchCapacity - seqLength, 64):
    
        X = np.zeros((batchSize, seqLength))
        Y = np.zeros((batchSize, seqLength, uniqueChars))
        
        for i in range(0,16):
            for j in range(0,64):
                X[i, j] = allChars[i * batchCapacity + s + j]
                Y[i, j, allChars[i * batchCapacity + s + j + 1]] = 1 
        yield X, Y
        
def build_model(batch_size, seq_length, unique_chars):
    model = Sequential()
    
    model.add(Embedding(input_dim = unique_chars, output_dim = 512, batch_input_shape = (batch_size, seq_length), name = "embd_1")) 
    
    model.add(LSTM(256, return_sequences = True, stateful = True, name = "lstm_1"))
    model.add(Dropout(0.2, name = "drp_1"))
    
    model.add(LSTM(256, return_sequences = True, stateful = True, name = "lstm_2"))
    model.add(Dropout(0.2,name = "drp_2"))
    
    model.add(LSTM(256, return_sequences = True, stateful = True, name = "lstm_3"))
    model.add(Dropout(0.2,name = "drp_3"))
    
    model.add(TimeDistributed(Dense(unique_chars)))
    model.add(Activation("softmax"))
    
    #TODO : load weights.
    #model.load_weights(os.path.join(model_weights_directory, 'Weights_1.h5'))

    return model

def train_model(data, epochs = 100):
    #mapping character to index number
    cti = {ch: i for (i, ch) in enumerate(sorted(list(set(data))))}
    print("Number of unique characters included in our ABC database = {}".format(len(cti)))
    #Will print the number of different charachters in our database
    
    with open(cij, mode = "w") as f:
        json.dump(cti, f)
        
    itc = {i: ch for (ch, i) in cti.items()}
    unique_chars = len(cti)
    
    model = build_model(batchSize, seqLength, unique_chars)
    model.summary()
    model.compile(loss = "categorical_crossentropy", optimizer = "RMSProp", metrics = ["accuracy"])
    
    all_chars = np.asarray([cti[c] for c in data], dtype = np.int32)
    print("Total number of characters = "+str(all_chars.shape[0])) #155222
    
    epoch_number, loss, accuracy = [], [], []
    
    for epoch in range(0, epochs):
        print("Epoch {}/{}".format(epoch+1, epochs))
        final_epoch_loss, final_epoch_accuracy = 0, 0
        epoch_number.append(epoch+1)
        
        for i, (x, y) in enumerate(makeBatch(all_chars, unique_chars)):
            final_epoch_loss, final_epoch_accuracy = model.train_on_batch(x, y) 
            print("Batch: {}, Loss: {}, Accuracy: {}".format(i+1, final_epoch_loss, final_epoch_accuracy))
        loss.append(final_epoch_loss)
        accuracy.append(final_epoch_accuracy)
        
        if (epoch + 1) % 1	 == 0:
            if not os.path.exists(model_weights_directory):
                os.makedirs(model_weights_directory)
            model.save_weights(os.path.join(model_weights_directory, "Weights_{}.h5".format(epoch+1)))
            print('Saved Weights at epoch {} to file Weights_{}.h5'.format(epoch+1, epoch+1))
    
    #creating dataframe and record all the losses and accuracies at each epoch Total 40
    log_frame = pd.DataFrame(columns = ["Epoch", "Loss", "Accuracy"])
    log_frame["Epoch"] = epoch_number
    log_frame["Loss"] = loss
    log_frame["Accuracy"] = accuracy
    log_frame.to_csv("log.csv", index = False)

if __name__ == "__main__":
    data_file = "Morris and Waltzes.txt"
    cij = "cti.json"
    model_weights_directory = 'Model_Weights1/'
    file = open(data_file, mode = 'r')
    data = file.read()
    file.close()
    train_model(data)