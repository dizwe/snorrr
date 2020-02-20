import os

from pydub import AudioSegment
import librosa
import numpy as np

def graph_spectrogram(wav_file, minus=True, nfft=2048, hop=512):
    rate, data = get_wav_info(wav_file)
    fs = rate # frequency
    
    S = librosa.feature.melspectrogram(data, sr=fs, n_mels=128, n_fft=nfft, hop_length=hop)
    log_S = librosa.power_to_db(S, ref=np.max)
    if not minus:
        return log_S
    
    data_minus = -data
    S_minus = librosa.feature.melspectrogram(data_minus, sr=fs, n_mels=128, n_fft=nfft, hop_length=hop)
    log_S_minus = librosa.power_to_db(S_minus, ref=np.max)
    return log_S, log_S_minus

# Load a wav file
def get_wav_info(wav_file):
    data, rate = librosa.load(wav_file, sr=44100)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio(audio_dir):
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir(audio_dir):
        if filename.endswith("wav"):
            if filename.startswith('background'):
                background = AudioSegment.from_wav(os.path.join(audio_dir, filename))
                backgrounds.append(background)
            elif filename.startswith('negative'):
                negative = AudioSegment.from_wav(os.path.join(audio_dir, filename))
                if len(negative) < 1000:
                    negatives.append(negative)
            else:
                activate = AudioSegment.from_wav(os.path.join(audio_dir, filename))
                activates.append(activate)
    return activates, negatives, backgrounds

# Load raw audio files for speech synthesis
def load_raw_audio_with_folder(audio_dir):
    activates = []
    backgrounds = []
    negatives = []
    for category in os.listdir(audio_dir): # folder별로 파일 정리
        category_dir = os.path.join(audio_dir, category)
        print(category_dir)
        if category == "background":
            for filename in os.listdir(category_dir): # folder
                if filename.endswith("mp3"):
                    background = AudioSegment.from_mp3(os.path.join(category_dir, filename))
                    backgrounds.append(background)
        elif category == "snore":
            for filename in os.listdir(category_dir): # folder
                if filename.endswith("mp3"):
                    activate = AudioSegment.from_mp3(os.path.join(category_dir, filename))
                    activates.append(activate)
        elif category == "negative":
            for filename in os.listdir(category_dir): # folder
                # negative는 librosa에서 cut 했는데 mp3로 저장했음에도 불구하고 wav로 되어있다.
                if filename.endswith("mp3"):
                    negative = AudioSegment.from_wav(os.path.join(category_dir, filename))
                    negatives.append(negative)
                    
    return activates, negatives, backgrounds

def make_beep_wav(wav, y, output_name):
    beep, _ = librosa.load('./check_sound/wow_long.wav', sr = 44100)
    data, _ = librosa.load(wav, sr=44100)

    length = y.shape[0]
    one_size = int(len(data)/length)
    for i in range(1, length):
        tmp = int(len(data)*(i)/length) # length 길이만큼 나누면 한 filter 크기를 나타내는거다!
        
        t_1 = y[i-1]
        t = y[i]
        
        if (t_1 == 0) and (t == 1): # 딱 그구간만 틀어주면 된다?
            # 이후 구간 1 개수 세기
            one_num =0
            while y[i]==1 and i<length-1:
                one_num = one_num+1
                i = i+1

            # 그 구간만큼 빼서 1넣기
            # mix하기 https://stackoverflow.com/questions/4039158/mixing-two-audio-files-together-with-python
            if tmp > one_num*one_size:
                # 테스트를 위해서는 실제 데이터와 beep데이터를 함께 들어야 하기 때문에 mix 하는것으로 수정
                data[tmp-one_num*one_size:tmp] = (beep[:one_num*one_size]*0.1+data[tmp-one_num*one_size:tmp])/2
                # data[tmp-one_num*one_size:tmp] = beep[:one_num*one_size]
            else:
                # 테스트를 위해서는 실제 데이터와 beep데이터를 함께 들어야 하기 때문에 mix 하는것으로 수정
                data[:tmp] =  (beep[:tmp]*0.1+data[:tmp])/2
        # if (t == 1): # 딱 그구간만 틀어주면 된다?
        #     print(tmp)
        #     if tmp > one_size :
        #         data[tmp-one_size:tmp] =  beep[:one_size]*0.5
        #     else:
        #         data[:tmp] =  beep[:tmp]*0.5
    librosa.output.write_wav(output_name, data, sr=44100)

def output_postprocessing(outputs, th):
    for output in outputs:
        output[output<th] = 0
        output[output>=th] = 1
    return outputs
