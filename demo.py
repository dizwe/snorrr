#%%
import os
import pandas as pd
import librosa
import numpy as np
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, multiply

from td_utils import *
from matplotlib import pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def make_model(input_shape):
    X_input = Input(shape = input_shape)
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)         # CONV1D
    X = BatchNormalization()(X)                                 # Batch normalization
    X = Activation('relu')(X)                                   # ReLu activation
    X = Dropout(0.2)(X)                                         # dropout (use 0.8)

    X = GRU(units = 128, return_sequences = True)(X)            # GRU (use 128 units and return the sequences)
    X = Dropout(0.2)(X)                                         # dropout (use 0.8)
    X = BatchNormalization()(X) # Batch normalization
    y = Activation("softmax")(X)
    X = multiply([y, X])
    

    X = GRU(units = 128, return_sequences = True)(X)            # GRU (use 128 units and return the sequences)
    X = Dropout(0.2)(X)                                         # dropout (use 0.8)
    X = BatchNormalization()(X) 
    y = Activation("softmax")(X)
    X = multiply([y, X])
    X = Dropout(0.2)(X)                                         # dropout (use 0.8)

    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)    # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X)
    return model  


def main2(file_name, idd=0):
    X = []
    ## 여기서 nfft hop이 다라서 이상하게 된거 같음
    X.append(np.transpose(graph_spectrogram(file_name, minus=False)))
    X = np.array(X)
    # print(X[0][0])
    # X = (X - X.min()) / (X.max() - X.min())
    print(bcolors.OKGREEN + '\nMade mel-spectrogram data ! ({})\n'.format(X.shape) + bcolors.ENDC)

    model = make_model(input_shape=(None, 128))
    model.load_weights("my_model/my_model_weight.h5")
    print(bcolors.OKGREEN + '\nMade model and Loaded weight ! \n' + bcolors.ENDC)

    # 이걸로 threshold 높이면 accuracy도 달라지지 않을까
    y = output_postprocessing(model.predict(X), 0.6)
    for i in range(len(y)):
        plt.plot(y[i], label='true')
        plt.show()

    print(bcolors.OKGREEN + 'Finished Predict ! \n' + bcolors.ENDC)

    make_beep_wav(file_name, y[0], "./demo/result_{}.wav".format(idd))
    print(bcolors.OKGREEN + 'Result: shape ({}) , # of 1 ({})\n'.format(y.shape, np.count_nonzero(y)) + bcolors.ENDC)
    print(bcolors.OKGREEN + '\nNow you can check!!! ' + bcolors.ENDC)

# 1 길이만큼 앞에 삐소리나게 하기
def main(file_name, idd=0):
    X = []
    ## 여기서 nfft hop이 다라서 이상하게 된거 같음
    X.append(np.transpose(graph_spectrogram(file_name, minus=False)))
    X = np.array(X)
    # print(X[0][0])
    # X = (X - X.min()) / (X.max() - X.min())
    print(bcolors.OKGREEN + '\nMade mel-spectrogram data ! ({})\n'.format(X.shape) + bcolors.ENDC)

    model = make_model(input_shape=(None, 128))
    model.load_weights("my_model/my_model_weight.h5")
    print(bcolors.OKGREEN + '\nMade model and Loaded weight ! \n' + bcolors.ENDC)

    # 이걸로 threshold 높이면 accuracy도 달라지지 않을까
    y = output_postprocessing(model.predict(X), 0.6)
    # for i in range(len(y)):
    #     plt.plot(y[i], label='true')
    #     plt.show()

    print(bcolors.OKGREEN + 'Finished Predict ! \n' + bcolors.ENDC)

    make_beep_wav(file_name, y[0], "./demo/result_{}.wav".format(idd))
    print(bcolors.OKGREEN + 'Result: shape ({}) , # of 1 ({})\n'.format(y.shape, np.count_nonzero(y)) + bcolors.ENDC)
    print(bcolors.OKGREEN + '\nNow you can check!!! ' + bcolors.ENDC)


#%%
if __name__ == '__main__':
    for file in os.listdir(os.path.join('.','test_data')):
        # 1599 mix_14 6055 8773 851 8847 
        if file.endswith('mp3'):
            file_name = os.path.join('.','test_data',file) #input('URL을 입력하세요: ')
            idd = file #int(input('ID: '))
            main(file_name, idd)

# %%
