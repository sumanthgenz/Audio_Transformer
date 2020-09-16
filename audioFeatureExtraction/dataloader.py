import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import openpyxl
import numpy as np
import os
import pickle
from collections import Counter



class AudioDataLoader(Dataset):
    
        def __init__(self, wav2LabelDictPath, wav2VectorDictPath):
            assert(wav2LabelDictPath != None)
            assert(wav2VectorDictPath != None)
            try :
                with open(wav2LabelDictPath, 'rb') as handle:
                    self.wav2LabelDict = pickle.load(handle)
                with open(wav2VectorDictPath, 'rb') as handle:
                    self.wav2VectorDict = pickle.load(handle)
            except :
                print("Error processing files: {}, {}".format(wav2LabelDictPath, wav2VectorDictPath))
            self.count = 0
            self.len = len(self.wav2LabelDict.keys())
            self.idx2wav = {}
            self.labels = []
            self.sizes = []
            for k, v in self.wav2VectorDict.items():
                self.sizes.append(v.shape[1])

            self.max_size =  np.max(self.sizes)

            for idx, wav in enumerate(self.wav2LabelDict.keys()):
                self.idx2wav[idx] = wav
                self.labels.append(self.wav2LabelDict[wav])
