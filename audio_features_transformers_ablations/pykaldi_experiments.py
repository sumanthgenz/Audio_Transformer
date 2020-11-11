import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger



import librosa
import openpyxl
import torch
import numpy
import numpy as np
import sklearn
# import surfboard
# from surfboard.sound import Waveform
# from surfboard.feature_extraction import extract_features
# from IPython.display import Audio
# import fastai
# import plotly
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from datetime import datetime
import warnings
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import pandas as pd 
from collections import Counter
import matplotlib.pyplot as plt
import wandb
import gc 

import kaldi
from kaldi.matrix import Matrix

from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.matrix import SubVector, SubMatrix
from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialWaveReader, RandomAccessWaveReader
from kaldi.util.table import MatrixWriter
from numpy import mean
from sklearn.preprocessing import scale
import kaldiio

from tqdm import tqdm

import torchaudio



wav_paths = []
rootdir = "/data3/kinetics_pykaldi/validate"
for path in glob.glob(f'{rootdir}/**/*.wav'):
    wav_paths.append(path)

# one_path = wav_paths[0]
# count = 1
# with open("kinetics_pykaldi_validate.scp", "w") as outpt:
#     for one_path in wav_paths:
#         outpt.write("{} {}\n".format(count, one_path))
#         count += 1


# usage = """Extract MFCC features.
#            Usage:  example.py [opts...] <rspec> <wspec>
#         """

# po = ParseOptions(usage)
# po.register_float("min-duration", 0.0,
#                   "minimum segment duration")
# mfcc_opts = MfccOptions()
# mfcc_opts.frame_opts.samp_freq = 22050
# mfcc_opts.register(po)

# opts = po.parse_args()

# rspec = "p,scp:kinetics_pykaldi_validate.scp" 

# mfcc = Mfcc(mfcc_opts)
# sf = mfcc_opts.frame_opts.samp_freq

# def kaldi_mfcc_extraction(wav):
#     pbar.update(1)

#     # s = wav.data()[:,::int(wav.samp_freq / sf)]

#     # # mix-down stereo to mono
#     # m = SubVector(mean(s, axis=0))

#     # # compute MFCC features
#     # a = np.array(mfcc.compute_features(m, sf, 1.0))

# utterances = []
# with SequentialWaveReader(rspec) as reader:
#     for key, wav in reader:
#         utterances.append(wav)
    
# batch_feats = Parallel(n_jobs=12, backend='loky')(delayed(kaldi_mfcc_extraction)(wav) for wav in utterances)

# print(np.array(batch_feats).shape)


# reader = RandomAccessWaveReader(rspec)
# batch_feats = []
# wav = reader['31959']
    
# s = wav.data()[:,::2]
# m = SubVector(mean(s, axis=0))
# f = np.transpose(np.array(mfcc.compute_features(m, sf, 1.0)))
# batch_feats.append(f)
pbar = tqdm(total=500)
trash = 0

# for w in wav_paths[0:500]:
#     try:
#         wav, samp_freq = torchaudio.load(w)
#         f = torchaudio.compliance.kaldi.mfcc(wav, sample_frequency=88200)
#         pbar.update(1)
#     except:
#         trash += 1
# print(trash)

wav, samp_freq = torchaudio.load('/data3/kinetics_pykaldi/validate/509_throwing axe/45->-oBtO8CxmzLk.wav')
f = np.transpose(torchaudio.compliance.kaldi.mfcc(wav, sample_frequency=88200))

print(f.shape)

# filePath = '/data3/kinetics_pykaldi/validate/509_throwing axe/45->-oBtO8CxmzLk.wav'
# (sig, rate) = librosa.load(filePath, offset=0, duration=10)
# mfcc = librosa.feature.mfcc(y=sig, sr=rate, n_mfcc=13)

# print("sig")
# print(wav.shape)
# print(sig.shape)
# print("rate")
# print(samp_freq)
# print(rate)
# print("mfcc")
# print(f.shape)
# print(mfcc.shape)


