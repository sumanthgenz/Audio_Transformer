import librosa
import surfboard
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
import tensorflow as tf
import openpyxl
import torch
import numpy
import sklearn
import numpy as np
import os
import pickle
import random


def dict2np(dic):
    return np.array(list(dic.values()))

def extract_mfcc_chroma(filePath):
    sound = Waveform(path=filePath, sample_rate=20000)
    mfcc = sound.mfcc()
    chroma = sound.chroma_cqt()
#     spec_entropy = sound.spectral_entropy()
#     spec_skew = sound.spectral_skewness()
#     return np.concatenate((mfcc, chroma, spec_entropy, spec_skew))
    return np.concatenate((mfcc, chroma))

def kld(p, q):
    return sum(p[i] * np.log(p[i]/(q[i]+0.1)) for i in range(len(p)))

def ang(x,y):
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    cos = np.dot(x, y)/(nx * ny)
    if cos > 1:
        cos = 1
    elif cos < -1:
        cos = -1
    return 1 - np.arccos(cos)/np.pi

def store_classes(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        classes.append(subdir)
    classes.remove(rootdir)
    return classes
    

def get_audio_features(rootdir):
    classes = store_classes(rootdir)
    labelDict = {}
    vectorDict = {}

    for subdir, dirs, files in os.walk(rootdir):
    for f in files:
        path = os.path.join(rootdir, subdir, f)
        try:
            count += 1
            labelDict[path] = classes.index(subdir)
            vectorDict[path] = extract_mfcc_chroma(path)
        except:
            print("bad file: " + path)
        if count % 1000 == 0:
            print("1000 done")
    return labelDict, vectorDict
            
def save_data(labelPath, vectorPath):
    with open('labelPath.pickle', 'wb') as handle:
        pickle.dump(labelPath, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('vectorPath.pickle', 'wb') as handle:
        pickle.dump(vectorPath, handle, protocol=pickle.HIGHEST_PROTOCOL)

rootdir = '/Users/sumanthgurram/Desktop/WAV'
store_classes(rootdir)
labels, vectors = get_audio_features(rootdir)
save_data(labels, vectors)


# vector = audio_extract("/Users/sumanthgurram/Desktop/WAV/SumoWrestling/v_SumoWrestling_g01_c01.wav").T
# vector = audio_extract("/Users/sumanthgurram/Desktop/WAV/PlayingCello/v_PlayingCello_g02_c05.wav").T
# vector = audio_extract("/Users/sumanthgurram/Desktop/WAV/CricketShot/v_CricketShot_g02_c05.wav").T
# vector = audio_extract("/Users/sumanthgurram/Desktop/WAV/Surfing/v_Surfing_g01_c05.wav").T
# vector = audio_extract("/Users/sumanthgurram/Desktop/WAV/SkyDiving/v_SkyDiving_g05_c02.wav").T

# cols = vector.shape[0]
# ang_dist = []
# rand_ang_dist= []
# baseline = ang(vector[25], vector[25])
# print(baseline)

# for i in range(cols-1):
#     try:
#         ang_dist.append(ang(vector[i], vector[i+1]))
#     except:
#         a = 1
# for i in range(cols):
#     todo = np.random.randint(cols, size=20)
#     for k in todo:
#         rand_ang_dist.append(ang(vector[i], vector[k]))
# print(kl_divs[0:25])
# print(np.average(rand_ang_dist))
# print(np.average(ang_dist))
