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

def populate_spec_savedir(savedir):
    for subfolder_name in classes:
        os.makedirs(os.path.join(savedir, subfolder_name))
    
def get_audio_features(rootdir, classes):
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

def get_mel_spec(rootdir, savedir, classes):
    spec2LabelDict = {}
    spec2VectorDict = {}
    count = 0
    for subdir, dirs, files in os.walk(rootdir):
        for f in files:
            wav_path = os.path.join(rootdir, subdir, f)
            
            #might need subdir.split("/")[-1]
            spec_path = os.path.join(savedir, subdir, f.split(".")[0])

            try:
                count += 1
                (sig, rate) = librosa.load(wav_path)
                fig = plt.figure(figsize=[0.5,0.5])
                ax = fig.add_subplot(111)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_frame_on(False)
                S = librosa.feature.melspectrogram(y=sig, sr=rate, n_mels=512)
                librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
                plt.savefig(spec_path, dpi=400, bbox_inches='tight',pad_inches=0)
                plt.close('all')
                
                #might need subdir.split("/")[-1]
                spec2LabelDict[spec_path] = classes.index(subdir.split("/")[-1])
                spec2VectorDict[spec_path] = np.array(Image.open(spec_path + ".png").convert('RGB').resize((150, 150)))/255
                
            except:
                print("bad file: " + wav_path)
            if count % 1000 == 0:
                print("1000 done")
    return spec2LabelDict, spec2VectorDict
            
def save_data(labels, vectors, labelPath, vectorPath):
    with open(labelPath + '.pickle', 'wb') as handle:
        pickle.dump(labelPath, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(vectorPath + '.pickle', 'wb') as handle:
        pickle.dump(vectorPath, handle, protocol=pickle.HIGHEST_PROTOCOL)

rootdir = 'WAV'
savedir = "UCF-101-SPEC"

classes = store_classes(rootdir)
populate_spec_savedir(savedir)
mfcc_labels, mfcc_vectors = get_audio_features(rootdir, classes)
spec_labels, spec_vectors = get_mel_spec(rootdir, savedir, classes)
save_data(mfcc_labels, mfcc_vectors, "mfccLabels", "mfccVectors")
save_data(spec_labels, spec_vectors, "specLabels", "specVectors")

# chmod +x convert.sh
# ./avitomp3.sh UCF-101/ WAV/
# ./mp3-wav.sh MP3/ WAV/
# ./convert.sh UCF-101/ WAV/


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
