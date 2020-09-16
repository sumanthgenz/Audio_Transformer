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
            self.num_elems = len(self.wav2LabelDict.keys())
            self.idx2wav = {}
            self.labels = []
            self.sizes = []
            for k, v in self.wav2VectorDict.items():
                self.sizes.append(v.shape[1])

            self.max_size =  np.max(self.sizes)

            for idx, wav in enumerate(self.wav2LabelDict.keys()):
                self.idx2wav[idx] = wav
                self.labels.append(self.wav2LabelDict[wav])

        def __len__(self):
            return self.num_elems

        def getNumClasses(self):
            return len(Counter(self.labels))

        def __getitem__(self, idx):
            wav = self.idx2wav[idx]
            vec = self.wav2VectorDict[wav]
            lab = self.wav2LabelDict[wav]
            to_pad = self.max_size - vec.shape[1]
            return vec, lab, self.max_size
