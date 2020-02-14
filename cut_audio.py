#%%
import os
import librosa
from matplotlib import pyplot as plt
# import numpy as np

data_path = os.path.join(os.getcwd(),'data','audio_label_clip','negative')

new_data_path = os.path.join(os.getcwd(),'data','audio_label_clip','negative_cut')
for file in os.listdir(data_path):
    wav_file = os.path.join(data_path, file)
    y,sr = librosa.load(wav_file)    
    total_time = len(y)/sr
    segment_num = 5 # 10초짜리를 몇개로 나눌것인가
    unit = int(len(y)/5)
    for i in range(segment_num):
        one_seg = y[unit*i:unit*(i+1)]
        librosa.output.write_wav(os.path.join(new_data_path,f'{file.replace(".mp3","")}_{i}.wav'), one_seg, sr)
    # time = np.linspace(0, len(y)/sr, len(y))
    # plt.plot(time, y)


#%%
import os
import librosa
from matplotlib import pyplot as plt
# import numpy as np

data_path = os.path.join(os.getcwd(),'data','audio_label_clip_for_test','negative')

new_data_path = os.path.join(os.getcwd(),'data','audio_label_clip_for_test','negative_cut')
for file in os.listdir(data_path):
    wav_file = os.path.join(data_path, file)
    y,sr = librosa.load(wav_file)    
    total_time = len(y)/sr
    segment_num = 5 # 10초짜리를 몇개로 나눌것인가
    unit = int(len(y)/5)
    for i in range(segment_num):
        one_seg = y[unit*i:unit*(i+1)]
        librosa.output.write_wav(os.path.join(new_data_path,f'{file.replace(".mp3","")}_{i}.wav'), one_seg, sr)
    # time = np.linspace(0, len(y)/sr, len(y))
    # plt.plot(time, y)



# %%
