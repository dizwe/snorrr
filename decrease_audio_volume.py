#%%
########
# backgorund 낮추기
# background 폴더 복사해서 original 폴더 따로 만들어 둘 것!
########
import os
from pydub import AudioSegment

data_path = os.path.join(os.getcwd(),'data','audio_label_clip','background')

new_data_path = os.path.join(os.getcwd(),'data','audio_label_clip','background_lower')
for file in os.listdir(data_path):
    song = AudioSegment.from_mp3(os.path.join(data_path, file))

    # reduce volume by 30 dB
    song_30_db_quieter = song - 30

    # # but let's make him *very* quiet
    # song = song - 36

    # save the output
    song_30_db_quieter.export(os.path.join(data_path,file), "mp3")

# %%
import os
from pydub import AudioSegment

data_path = os.path.join(os.getcwd(),'data','audio_label_clip','background')

# new_data_path = os.path.join(os.getcwd(),'data','audio_label_clip','snore_lower')
for file in os.listdir(data_path):
    if file.endswith('mp3'):
        song = AudioSegment.from_mp3(os.path.join(data_path, file))

        # reduce volume by 10 dB
        song_db_quieter = song - 10

        # # but let's make him *very* quiet
        # song = song - 36

        # save the output
        song_db_quieter.export(os.path.join(data_path,file), "mp3")


# %%
