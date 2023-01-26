import numpy as np
from pydub import AudioSegment
import random
import sys
import io 
import os


import matplotlib.pyplot as plt
from scipy.io import wavfile



def graph_spectrogram(wav_file):
    rate, data = wavfile.read(wav_file)
    lf = 200 #length of windows segment
    fs = 8000 #frequency of sampling
    noverlap = 120 #overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, lf, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:0], lf, fs, noverlap=noverlap)
    return pxx


def match_target_amplitude(sound, targetdBFS):
    change_in_dBFS = targetdBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def load_raw_audio(path):
    dic = {} #on off to switch ac in lift
    folders = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'up', 'down', 'yes', 'no', 'on', 'off', 'stop']
    for trig in folders:
        dic[trig] = []
    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'up', 'down', 'yes', 'no', 'on', 'off', 'stop']
    for i in range(len(words)):
        for filename in os.listdir(path+words[i]):
            if filename.endswith('wav'):
                temp = AudioSegment.from_wav(path+words[i]+'/'+filename)
                dic[folders[i]].append(temp)
    return dic


# #preprocess raw backgrounds to size 2400
# def create_backgroundsmls(backgroundsall, output_path='C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/train/audio/_backs_smls_/', size=2400, stride=1000):
#     #merges raw backgrounds into one then processes to size and each cut with window=stride
#     #backgroundsall list = dic['background']
#     merged = backgroundsall[0]
#     backgroundsmls = []
#     for i in range(1, len(backgroundsall)):
#         merged+=backgroundsall[i]
#     for i in range(0, len(merged)-(size+2), stride):
#         temp = merged[i+1:i+size+1]
#         temp.export(output_path + str(i//stride) + '.wav', format='wav')
#         backgroundsmls.append(temp)
#     return backgroundsmls

def load_raw_file(path):
    temp = AudioSegment.from_wav(path)
    return temp


raw_data = load_raw_audio('C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/train/audio/')

#this function gets random time segments to insert trigger words in background given segmentms (len(seg)) in ms
def get_timeseg(segmentms):
    segment_start = np.random.randint(low=0, high=2400-segmentms)
    segment_end = segment_start + segmentms - 1
    return (segment_start, segment_end)


#this function returns if there is a overlap between previously inserted files and next insertion
def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time
    overlap = False
    for previous_start, previous_end in previous_segments:
        if (previous_start<=segment_end and previous_start>=segment_start) or (segment_start<-previous_end and segment_start>=previous_start):
            overlap = True
            break
    return overlap


#this function inserts a single trigger file into a background
def insert_trigr(backgroundsml, trigr, previous_segments):
    segmentms = len(trigr)
    segment_time = get_timeseg(segmentms)
    retry = 5
    while is_overlapping(segment_time, previous_segments) and retry>0:
        segment_time = get_timeseg(segmentms)
        retry-=1
    if not is_overlapping(segment_time, previous_segments):
        previous_segments.append(segment_time)
        new_background = backgroundsml.overlay(trigr, position=segment_time[0])
    else:
        new_background = backgroundsml
        segment_time = (2400, 2400)
    return new_background, segment_time


#this function labels the example for training
def insert_label(y, segment_endms, class_no, labels=10):
    _, Ty = y.shape
    segment_end_y = int(segment_endms*Ty/2400.0) #duration of background in terms of spectogram time_steps
    if segment_end_y < Ty:
        for i in range(segment_end_y+1, segment_end_y+labels):
            if i<Ty:
                y[0, i] = class_no
    return y


# '''
# tets = graph_spectrogram('C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/train/audio/_backs_smls_/0.wav')
# n_freq, Tx = tets.shape
# '''
Ty = 128 # earlier 275, these hyperparams can be changed



backgroundsmls = []
for filename in os.listdir('C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/train/audio/_backs_smls_'):
    if filename.endswith('wav'):
        temp = AudioSegment.from_wav('C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/train/audio/_backs_smls_'+'/'+filename)
        backgroundsmls.append(temp)

        
#here a single training example is created
def create_training_ex(backgroundsml, trigs, trig_no, Ty=128):
    #Ty is number of units in which output audio will be devided
    #trigs = bed, bird, four... files in pydub
    backgroundsml = backgroundsml - 20
    y = np.zeros((1,Ty))
    previous_segments = []
    num_trigrs = np.random.randint(0, 2)
    random_indices = np.random.randint(len(trigs), size=num_trigrs)
    random_trigs = [trigs[i] for i in random_indices]
    for random_trig in random_trigs:
        backgroundsml, segment_time = insert_trigr(backgroundsml, random_trig, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_label(y, segment_end, trig_no)
    background = match_target_amplitude(backgroundsml, -20.0)
    file_handle = background.export('train'+'.wav', format='wav')
    x = graph_spectrogram('train.wav')
    return x, y


#here finally we create a whole training set
def create_fulltrainingset(dic, backgroundsmls, Ty=128):
    #dic without background
    np.random.seed(466)
    #X = []
    #Y = []
    lbl = 1
    for i in dic:
        for j in range(min(len(dic[str(i)]),2000)): #len(dic[str(i)])
            if j%100==0:
                print(j)
            k = np.random.randint(0, len(backgroundsmls)-1)
            x, y = create_training_ex(backgroundsmls[k], dic[str(i)][j], lbl, Ty)
            #X.append(np.array(x.swapaxes(0,1)))
            #Y.append(np.array(y.swapaxes(0,1)))
            np.save(f'C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/X_train/x{str(i)}-{str(j)}.npy', np.array(x.swapaxes(0,1)))
            np.save(f'C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/Y_train/y{str(i)}-{str(j)}.npy', np.array(y.swapaxes(0,1)))
        lbl+=1
    # '''
    # #to save
    # #np.save(f'C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/XY_train/X.npy', X)
    # #np.save(f'C:/Users/2jeet/Desktop/Sujuz/ml_projects/4. trigger_wordsss/XY_train/Y.npy', Y)
    # '''
    #return X, Y

# def save_full_trainingset(dic, backgroundsmls, Ty=128, save_dir="./"):
#     save_x = os.path.join(save_dir, "X.txt")
#     save_y = os.path.join(save_dir, "Y.txt")
#     for i in dic:
#         for j in range(2000): #len(dic[str(i)])
#             if j%100==0:
#                 print(j)
#             k = np.random.randint(0, len(backgroundsmls)-1)
#             x, y = create_training_ex(backgroundsmls[k], dic[str(i)][j], lbl, Ty)
#             print(x)
    


create_fulltrainingset(raw_data, backgroundsmls)

print('Successfully created!!! Now check!')
