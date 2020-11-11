import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
import pickle
from collections import Counter
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(name='Remote_Audio_Transformer',project='audioFeatureExtraction')
import wandb




import librosa
# import tensorflow as tf
import openpyxl
import torch
import numpy
import sklearn
# import surfboard
# from surfboard.sound import Waveform
# from surfboard.feature_extraction import extract_features
import numpy as np
import os
import pickle

print(torch.__version__)
print(pl.__version__)

class Net(pl.LightningModule):

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
            self.Labels = []
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

    def __init__(self):
        super(Net, self).__init__()
        self.num_features = 25
        self.input_size = 128
        self.seq_len = 500
        self.hidden_size = 80
        self.other_hidden_size = 80
        self.num_classes = 51
        self.num_epochs = 50
        self.batch_size = 80
        self.learning_rate = 0.00018
        self.attention_heads = 4
        self.n_layers = 4

        self.wav2labelPath = '/home/sgurram/Desktop/wav2LabelDict.pickle'
        self.wav2vecPath = '/home/sgurram/Desktop/wav2VectorDict.pickle'
        self.loss = nn.CrossEntropyLoss()

        self.fc0 = nn.Linear(self.num_features, self.input_size)
        self.encode = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.attention_heads)
        self.transformer = nn.TransformerEncoder(self.encode, num_layers=self.n_layers)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size) 
        self.fc2 = nn.Linear(self.hidden_size, self.other_hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(self.other_hidden_size, self.num_classes)
        self.train_dataset, self.test_dataset = self.prepare_data()
    
    def forward(self, x, mask):
        # expand = self.fc0(x)
        # transform = self.transformer(expand, src_key_padding_mask=mask)
        # mean = self.fc1(transform.permute(1, 0, 2).mean(dim=1))
        # rel = self.relu(mean)
        # next = self.fc2(rel)
        # rel2 = self.relu(next)
        # out = self.fc3(rel2)

        expand = self.fc0(x)
        transform = self.transformer(expand, src_key_padding_mask=mask)
        mean = self.fc1(transform.permute(1, 0, 2).mean(dim=1))
        rel = self.relu(mean)
        out = self.fc3(rel)
        return out

    def prepare_data(self):
        full_dataset = self.AudioDataLoader(self.wav2labelPath, self.wav2vecPath)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        return torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    def collate_fn(self, batch):
        def trp(arr, n):
            arr = arr.tolist()
            seq = []
            for cmp in arr:
                new_cmp = cmp[:n] + [0.0]*(n-len(cmp))
                seq.append(new_cmp)
            result = np.transpose(np.array(seq)).tolist()
            return result

        def pad_masker(arr):
            zer = torch.zeros(self.num_features)
            ind = 0
            for t in arr:
                pad = torch.all(torch.eq(t, zer))
                if  pad == 1:
                    break
                ind += 1
            result = ([True]*ind) + ([False]*(self.seq_len - ind))
            return result
        
        batch = np.transpose(batch)
        labels = batch[1].tolist()

        samples = torch.Tensor([(trp(t, self.seq_len))for t in batch[0]])
        mask = torch.Tensor([pad_masker(t) for t in samples])
        labels = torch.Tensor(labels).long()
        samples = samples.permute(1, 0, 2)

        return samples, labels, mask

    def train_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=True,
                                                        collate_fn=self.collate_fn)
        return self.train_loader


    def val_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=True,
                                                        collate_fn=self.collate_fn)        
        return self.test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target, mask = batch
        mask = mask < 0
        output = self.forward(data, mask)

        loss = self.loss(output, target)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        data, target, mask = batch
        mask = mask < 0
        output = self.forward(data, mask)
        temp_loss = self.loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = correct/self.batch_size
        logs = {'val_loss': temp_loss, 'acc': acc}
        return {'val_loss': temp_loss, 'acc': acc}

if __name__ == '__main__':
    model = Net()
    # trainer = pl.Trainer(gpus=4, max_epochs=2, logger=wandb_logger)
    trainer = pl.Trainer(gpus=[0, 1, 2, 3], max_epochs=50, logger=wandb_logger, distributed_backend='ddp')
    trainer.fit(model)
    trainer.save_checkpoint("audioTransformer5.ckpt")
