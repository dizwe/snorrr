import soundfile as sf
import subprocess
import random
import librosa
import numpy as np
from pydub import AudioSegment
import os

#%%
def wav_to_mp3(current_dir,filename):
    print(filename)
    subprocess.call(['ffmpeg','-i',
            os.path.join(current_dir,f'{filename}.wav'),
            os.path.join(current_dir,f'{filename}.mp3')
        ])
    subprocess.call(['rm',os.path.join(current_dir,f'{filename}.wav')])

#%%
def adding_random_noise(source_dir, filename, sr=22050, noise_rate=0.01):
    data, sr = librosa.load(os.path.join(source_dir,filename), sr=sr)
    # We limited the amplitude of the noise so we can still hear the word even with the noise, 
    # which is the objective
    rn = np.random.randn(len(data))
    data_rn = data + noise_rate*rn
    new_filename = f'{filename.replace(".mp3","")}_random'
    sf.write(os.path.join(source_dir,f'{new_filename}.wav'), data_rn, sr)
    wav_to_mp3(source_dir, new_filename)
    # bad sample 오류 뜸.
    # librosa.output.write_wav(os.path.join(audio_dir,'white_noise.wav'), data_rn, sr=sr) 
    # mp3로 변형하기

    print('random Noise 저장 성공')

#%%
def adding_white_noise(source_dir, filename, sr=22050, noise_rate=0.005):
    sound_file = AudioSegment.from_file(os.path.join(source_dir,filename))
    # 길이 만큼 random 돌려서 걔 얻기
    file_list = os.listdir(noise_dir)
    noise_file_idx = random.randint(0,len(file_list)-1)
    noise_file = AudioSegment.from_file(os.path.join(noise_dir,file_list[noise_file_idx]))
    noise_file = noise_file-25
    new_sound = sound_file.overlay(noise_file)
    new_sound.export(os.path.join(source_dir,f'{filename}_white.mp3'),format='mp3')
    print('White Noise 저장 성공')

    return 0

#%%
def shifting_sound(source_dir, filename, sr=22050, roll_rate=0.3):
    data, sr = librosa.load(os.path.join(source_dir,filename), sr=sr)
    # 그냥 [1, 2, 3, 4] 를 [4, 1, 2, 3]으로 만들어주는건데 이게 효과있는지는 잘 모르겠
    data_roll = np.roll(data, int(len(data) * roll_rate))
    new_filename = f'{filename.replace(".mp3","")}_shift'
    librosa.output.write_wav(os.path.join(source_dir,f'{new_filename}.wav'), data_roll, sr=sr)
    wav_to_mp3(source_dir, new_filename)
    print('rolling_sound 저장 성공')

#%%
def stretch_sound(source_dir, filename, sr=22050, rate=0.7):
    data, sr = librosa.load(os.path.join(source_dir,filename), sr=sr)
    # stretch 해주는거 비율이 뭐가 좋은지 잘모르겟, 0.8이랑, 1.2랑 차이가 안나는거 같음
    stretch_data = librosa.effects.time_stretch(data, rate)
    new_filename = f'{filename.replace(".mp3","")}_stretch'
    librosa.output.write_wav(os.path.join(source_dir,f'{new_filename}.wav'), stretch_data, sr=sr)
    wav_to_mp3(source_dir, new_filename)
    print('stretch_data 저장 성공')

#%%
def reverse_sound(source_dir,filename, sr=22050):
    data, sr = librosa.load(os.path.join(source_dir,filename), sr=sr)
    temp_array = []
    for i in range(len(data)):
        temp_array.append(data[len(data)-1-i])
    temp_numpy =np.asarray(temp_array)
    new_filename = f'{filename.replace(".mp3","")}_reverse'
    librosa.output.write_wav(os.path.join(source_dir,f'{new_filename}.wav'), temp_numpy, sr=sr)
    wav_to_mp3(source_dir, new_filename)
    print('reverse_data 저장 성공')
#%%
# 원래파일이랑 거의 똑같이 들림
def minus_sound(source_dir,filename, sr=22050):
    data, sr = librosa.load(os.path.join(source_dir,filename), sr=sr)
    temp_numpy = (-1)*data
    new_filename = f'{filename.replace(".mp3","")}_minus'
    librosa.output.write_wav(os.path.join(source_dir,f'{new_filename}.wav'), temp_numpy, sr=sr)
    wav_to_mp3(source_dir, new_filename)
    print('minus_data 저장 성공')


#%%
def freq_augmentation(source_dir,filename, sr=22050):
    data, sr = librosa.load(os.path.join(source_dir,filename), sr=sr)
    switch = np.random.randint(1,3)
    # if(switch==0): #원본
    #     return [data,sr]
    if(switch==1): #고음
        rate = np.random.uniform(2, 5)
    elif(switch==2): #저음
        rate = np.random.uniform(-5, -2)
    y = librosa.effects.pitch_shift(data, sr, n_steps=rate)
    new_filename = f'{filename.replace(".mp3","")}_freq'
    librosa.output.write_wav(os.path.join(source_dir,f'{new_filename}.wav'), y, sr=sr)
    wav_to_mp3(source_dir, new_filename)
    print('freq_data 저장 성공')



#%%
import random

#%%
audio_dir = os.path.join('..','data','audio_label_clip')
noise_dir = os.path.join('.','white_noise')

#%%
label_list = ['Male speech, man speaking','Outside, rural or natural','snoring','Traffic noise, roadway noise','Vehicle']
func_list = [adding_random_noise,adding_white_noise,shifting_sound,stretch_sound,reverse_sound,minus_sound,freq_augmentation]

for one_folder in os.listdir(audio_dir):
    if one_folder in label_list:
        print(one_folder)
        source_dir = os.path.join(audio_dir, one_folder)
        for file in os.listdir(source_dir):
            if file.endswith('mp3'):
                func_idx = random.randint(0,len(func_list)-1)
                func_list[func_idx](source_dir, file)
