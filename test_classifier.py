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

#%%
import copy
def anticipate_snore(file_name):
    X = []
    ## 여기서 nfft hop이 다라서 이상하게 된거 같음
    X.append(np.transpose(graph_spectrogram(file_name, minus=False)))
    X = np.array(X)
    # print(bcolors.OKGREEN + '\nMade mel-spectrogram data ! ({})\n'.format(X.shape) + bcolors.ENDC)

    model = make_model(input_shape=(None, 128))
    model.load_weights("my_model/my_model_weight.h5")
    # print(bcolors.OKGREEN + '\nMade model and Loaded weight ! \n' + bcolors.ENDC)

    # 이걸로 threshold 높이면 accuracy도 달라지지 않을까
    pred = model.predict(X)
    # ref라 값이 바뀜 -> deep copy
    pred_with_prob = copy.deepcopy(pred)
    y = output_postprocessing(pred, 0.9)

    # for i in range(len(y)):
    #     plt.plot(y[i], label='true')
    #     plt.show()

    # print(bcolors.OKGREEN + 'Finished Predict ! \n' + bcolors.ENDC)
    # print('filename',file_name,'len', len(y[y==1])>0)
    print(bcolors.OKGREEN + '\nNow you can check!!! ' + bcolors.ENDC)
    return len(y[y==1])>0, len(y[y==1]), len(y), pred_with_prob


#%%
import os
import pandas as pd

df = pd.DataFrame([],columns = ['data','pred','whole','one_num','pred_with_prob','real'])
for folder in os.listdir(os.path.join('.','test_data')):
    # 90% 이상의 확률로 맞을거라고 예측한 애가 얼마나 되는지 확인.
    if folder=="snore":
        for i, file in enumerate(os.listdir(os.path.join('.','test_data',folder))):
            if file.endswith('mp3'):
                print(i)
                file_name = os.path.join('.', 'test_data', folder, file) 
                pred, one_num, whole, pred_with_prob = anticipate_snore(file_name)
                df= df.append({'data':file, 'pred':pred, 'whole' : whole,'one_num':one_num,'pred_with_prob':pred_with_prob,'real':True},ignore_index=True)
                
    elif folder=="not_snore":
        for i, file in enumerate(os.listdir(os.path.join('.','test_data',folder))):
            if file.endswith('mp3'):
                print(i)
                file_name = os.path.join('.', 'test_data', folder, file) 
                pred, one_num, whole, pred_with_prob = anticipate_snore(file_name)
                df= df.append({'data':file, 'pred':pred, 'whole' : whole,'one_num':one_num, 'pred_with_prob':pred_with_prob,'real':False},ignore_index=True)

# %%
len(df[df['pred']==df['real']])/len(df)

#%%
# 0.572463768115942밖에 안나옴...
df.loc[274,'pred_with_prob']>0.5

# %%
# 그냥 to_csv로 저장하면 Numpy object로 저장한거 바꾸는데 애를 먹는다.
# df.to_csv('test_result.csv')
# k = pd.read_csv('test_result.csv')
df.to_pickle('test.csv')
results = pd.read_pickle('test.csv')

# %%
# 원래는 1이 하나라도 있으면 true라고 했는데 얼마 정도 해야 1으로 판단하면 좋을까?
# 몇개 이상 코골이라고 해야 true라고 하기
for threshold in range(0, 60, 10):
    print(f'-----threshold {threshold}')
    for p in range(50,100,5):
        results[f'pred_with_prob_{p}'] = results['pred_with_prob'].apply(lambda x: x>0.01*p).apply(lambda x: len(x[x==True])>threshold)
        print(f'{p}%')
        print(len(results[results[f'pred_with_prob_{p}']==results['real']])/len(results))
# %%
# 전체 크기가 190~210(1초에 약 20) 사이로 다양하다
len(results.loc[3,'pred_with_prob'][0])

# %%
