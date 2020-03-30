#%%
import os
import librosa
from matplotlib import pyplot as plt
import subprocess
# import numpy as np

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
data_path = os.path.join(parent_dir,'data','home')

new_data_path = os.path.join(parent_dir,'data','home_cut')

#%%
######### librosa cut
def wav_to_mp3(current_dir,filename):
    print(filename)
    subprocess.call(['ffmpeg','-i',
            os.path.join(current_dir,f'{filename}.wav'),
            os.path.join(current_dir,f'{filename}.mp3')
        ])
    subprocess.call(['rm',os.path.join(current_dir,f'{filename}.wav')])

for file in os.listdir(data_path):
    if file.endswith(".mp3"):
        wav_file = os.path.join(data_path, file)
        y,sr = librosa.load(wav_file)    
        total_time = len(y)/sr
        # segment_num = 5 # 10초짜리를 몇개로 나눌것인가
        # unit = int(len(y)/5)
        unit = 10*sr
        segment_num = len(y)/unit # 몇개의 10초로 자를것인가
        for i in range(segment_num):
            print(i)
            one_seg = y[unit*i:unit*(i+1)]
            new_file_name  = f'{file.replace(".mp3","")}_{i}.wav'
            librosa.output.write_wav(os.path.join(new_data_path, new_file_name), one_seg, sr)
            wav_to_mp3(new_data_path, new_file_name)
        # time = np.linspace(0, len(y)/sr, len(y))
        # plt.plot(time, y)



# %%
########## pydub cut
from pydub import AudioSegment

for file in os.listdir(data_path):
    if file.endswith(".mp3"):
        print(file)
        aud_file = AudioSegment.from_file(os.path.join(data_path,file))
        start =0
        ten_seconds = 10 * 1000
        cut_range = 240 # 40분
        
        for i in range(cut_range):
            print(i)
            start = start+ 10 * 1000
            cut_file = aud_file[start:start +10 * 1000]
            cut_file.export(os.path.join(new_data_path, f"{i}_{file}"), format="mp3")



# %%
