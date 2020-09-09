import torch
from torch.utils.data import Dataset, DataLoader
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
import surfboard
import openpyxl
import torch
import numpy
import sklearn
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
from collections import Counter
import numpy as np
import os
import pickle

from dataloader import AudioDataLoader, custom_collate

class Net(pl.LightningModule):

    def __init__(self):
        super(Net, self).__init__()
        self.num_features = 25
        self.input_size = 128
        self.seq_len = 500
        self.hidden_size = 80
        self.other_hidden_size = 80
        self.num_classes = 51
        self.num_epochs = 200
        self.batch_size = 60
        self.learning_rate = 0.00025
        self.attention_heads = 4
        self.n_layers = 4

        # self.wav2labelPath = '/home/sgurram/Desktop/wav2LabelDict.pickle'
        # self.wav2vecPath = '/home/sgurram/Desktop/wav2VectorDict.pickle'
        self.wav2labelPath = '/Users/sumanthgurram/Desktop/Audio_Feature_Transformer/wav2LabelDict.pickle'
        self.wav2vecPath = '/Users/sumanthgurram/Desktop/Audio_Feature_Transformer/wav2VectorDict.pickle'
        self.loss = nn.CrossEntropyLoss()

        self.fc0 = nn.Linear(self.num_features, self.input_size)
        self.encode = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.attention_heads)
        self.transformer = nn.TransformerEncoder(self.encode, num_layers=self.n_layers)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size) 
        self.fc2 = nn.Linear(self.hidden_size, self.other_hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(self.other_hidden_size, self.num_classes)
    
    def forward(self, x, mask):
        expand = self.fc0(x)
        transform = self.transformer(expand, src_key_padding_mask=mask)
        mean = self.fc1(transform.permute(1, 0, 2).mean(dim=1))
        rel = self.relu(mean)
        out = self.fc3(rel)
        return out

    def prepare_data(self):
        full_dataset = AudioDataLoader(self.wav2labelPath, self.wav2vecPath)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    

    def collate_fn(self, batch):
        return custom_collate(batch, self.num_features, self.seq_len)


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

    def training_step(self, batch, batch_idx):
        data, target, mask = batch
        mask = mask < 0
        output = self.forward(data, mask)

        loss = self.loss(output, target)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        wandb.init()
        data, target, mask = batch
        mask = mask < 0
        output = self.forward(data, mask)
        temp_loss = self.loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = torch.tensor(correct/self.batch_size)
        logs = {'val_loss': temp_loss, 'val_acc': acc}
        return {'val_loss': temp_loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([m['val_loss'] for m in outputs]).mean()
        avg_acc = torch.stack([m['val_acc'] for m in outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': logs}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == '__main__':
    model = Net()
    path_to_save = '/Users/sumanthgurram/Desktop/Audio_Feature_Transformer/'
    trainer = pl.Trainer(default_root_dir=path_to_save, gpus=0, max_epochs=5, logger=wandb_logger)
    trainer.fit(model)
