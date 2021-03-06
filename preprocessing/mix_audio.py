import os
import numpy as np
from td_utils import graph_spectrogram
from td_utils import graph_melspectrogram

def get_random_time_segment(segment_ms, total_ms=10000.0):
    # 넣을 segment 범위 얻기
    segment_start = np.random.randint(low=0, high=total_ms-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)

def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time
    overlap = False
    # 이미 넣은 segment 범위 있는지 확인
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap

def insert_audio_clip(background, audio_clip, previous_segments):
    total_ms = len(background)
    segment_ms = len(audio_clip)
    # 실제로 segment 삽입할 위치를 찾기
    segment_time = get_random_time_segment(segment_ms, total_ms)

    count = 0 
    # 이미 삽입된건 아닌지 previous_segments 확인하고 집어넣기(계속 시도해보고 안되면 그냥 포기)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms, total_ms)
        count += 1
        if count > 50 :
            return background, None

    # 이제 넣었으니까 집어넣기
    previous_segments.append(segment_time)
    # 그냥 그부분을 덮어쓰는걸로 수정해보자
    new_background = background[:segment_time[0]]+audio_clip+background[segment_time[1]:]
    
    return new_background, segment_time

def insert_audio_clip_with_overlay(background, audio_clip, previous_segments):
    total_ms = len(background)
    segment_ms = len(audio_clip)
    # 실제로 segment 삽입할 위치를 찾기
    segment_time = get_random_time_segment(segment_ms, total_ms)

    count = 0 
    # 이미 삽입된건 아닌지 previous_segments 확인하고 집어넣기(계속 시도해보고 안되면 그냥 포기)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms, total_ms)
        count += 1
        if count > 50 :
            return background, None

    # 이제 넣었으니까 집어넣기
    previous_segments.append(segment_time)
    # pydub에 overlay 기능
    new_background = background.overlay(audio_clip, position = segment_time[0])
    
    return new_background, segment_time

def insert_ones(y, segment_time, total_ms=10000.0):
    # 전체 표시할 길이
    Ty = y.shape[1]
    # segment_start = segment_time[0]
    # segment_end = segment_time[1]
    segment_end_y = int(segment_time[1] * Ty / total_ms)
    # total_ms에서 segment_end-segment_start 만큼의 길이 비만큼 Ty를 사용함
    segment_len = int((segment_time[1]-segment_time[0]) * Ty / total_ms)
    for i in range(segment_end_y + 1, segment_end_y + segment_len + 1):
        if i < Ty: # Ty가 전체 길이인데 넘어가면 안되니까 미연에 방지
            y[0, i] = 1
    return y

def create_training_data(background, activates, negatives, filename, kernel=15, stride=4, use_mel=False):
    background.export('../data/tmp.wav', format="wav")
    if use_mel==True:
        mel = graph_melspectrogram('../data/tmp.wav', minus=False)
    else:
        mel = graph_spectrogram('../data/tmp.wav', minus=False)
    Ty = int((mel.shape[1]-kernel)/stride + 1)

    y = np.zeros((1, Ty))
    total_ms = len(background)
    previous_segments = []
    input_ms = 0

    # 얼마 비율 이상 원하는 negative를 넣을 것인가?(아니면 계속 반복됨)
    while((input_ms/total_ms)<0.3):
        # 몇개를 집어넣을거냐! 0,2로 해서 빈도 좀 줄이기(background 더 자주 들을 수 있게)
        number_of_activates = np.random.randint(1, 2)
        # print('number_of_activates2',number_of_activates)
        # 파일들 중에서 랜덤으로 뽑기
        random_indices = np.random.randint(len(activates), size=number_of_activates)
        random_activates = [activates[i] for i in random_indices]
        
        for random_activate in random_activates:
            # ?? 뒤에 random 숫자는 왜 넣는거지?
            random_activate += np.random.randint(-2,5)
            background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
            if segment_time is not None: 
                segment_start, segment_end = segment_time
                input_ms += (segment_end - segment_start)
                y = insert_ones(y, segment_time=segment_time, total_ms=total_ms)

        # 사람 목소리는 더 빈도가 적게 나오게 하자(0,1)
        number_of_negatives = np.random.randint(0, 2)
        # print('number_of_negatives2',number_of_negatives)
        random_indices = np.random.randint(len(negatives), size=number_of_negatives)
        random_negatives = [negatives[i] for i in random_indices]

        for random_negative in random_negatives:
            random_negative += np.random.randint(-2,5)
            background, segment_time = insert_audio_clip(background, random_negative, previous_segments)
            if segment_time is not None:
                segment_start, segment_end = segment_time
                input_ms += (segment_end - segment_start)
                
    file_handle = background.export(filename, format="wav")
    if use_mel==True:
        x, x_minus = graph_melspectrogram(filename)
    else:
        x, x_minus = graph_spectrogram(filename)
    return x, x_minus, y
