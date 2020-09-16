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
                raise Exception('Invalid path')
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
                self.Labels.append(self.wav2LabelDict[wav])

        def __len__(self):
            return self.len

        def getNumClasses(self):
            return len(Counter(self.Labels))

        def __getitem__(self, idx):
            wav = self.idx2wav[idx]
            vec = self.wav2VectorDict[wav]
            lab = self.wav2LabelDict[wav]
            to_pad = self.max_size - vec.shape[1]
            return vec, lab, self.max_size

def custom_collate(batch, num_features, seq_len):
        def trp(arr, n):
            arr = arr.tolist()
            seq = []
            for cmp in arr:
                new_cmp = cmp[:n] + [0.0]*(n-len(cmp))
                seq.append(new_cmp)
            result = np.transpose(np.array(seq)).tolist()
            return result

        def pad_masker(arr):
            zer = torch.zeros(num_features)
            ind = 0
            for t in arr:
                pad = torch.all(torch.eq(t, zer))
                if  pad == 1:
                    break
                ind += 1
            result = ([True]*ind) + ([False]*(seq_len - ind))
            return result
        
        batch = np.transpose(batch)
        labels = batch[1].tolist()

        samples = torch.Tensor([(trp(t, seq_len))for t in batch[0]])
        mask = torch.Tensor([pad_masker(t) for t in samples])
        labels = torch.Tensor(labels).long()
        samples = samples.permute(1, 0, 2)
        return samples, labels, mask
