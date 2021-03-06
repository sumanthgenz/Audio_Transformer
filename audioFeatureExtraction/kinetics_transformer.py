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
import surfboard
import openpyxl
import torch
import numpy
import numpy as np
import sklearn
from surfboard.sound import Waveform
from surfboard.feature_extraction import extract_features
import librosa
import librosa.display
from IPython.display import Audio
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from datetime import datetime
import warnings
import glob
from tqdm import tqdm
import fastai
from joblib import Parallel, delayed
import pickle
import pandas as pd 
from collections import Counter
import matplotlib.pyplot as plt
import plotly
import wandb
import gc 


wandb_logger = WandbLogger(name='Remote_Audio_Transformer',project='kinetics-audio-feature-extraction')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# gc.collect()
# torch.cuda.empty_cache()

class Net(pl.LightningModule):

    class AudioDataLoader(Dataset):
    
        def __init__(self, csvType):
            try:
                self.df = pd.read_csv("/home/sgurram/Desktop/{}.csv".format(csvType))[0:100]
            except:
                print("Error processing csv for {}".format(csvType))
            self.classes = self.get_pickle("classes_dict")
            self.num_classes = len(list(self.classes.keys()))

        def get_pickle(self, classPath):
            with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
                result = pickle.load(handle)
            return result
        
        def __len__(self):
            return len(self.df)

        def getNumClasses(self):
            return self.num_classes

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            warnings.filterwarnings("ignore")

            name = row['youtube_id']
            start = row['time_start']
            end = row['time_end']
            split = row['split']
            label = "test_NAN_label"
            num_label = 999

            filePath = '/data2/kinetics/'

            if (split == "train"):
                filePath += 'kinetics_train/'
                label = row['label']
                num_label = self.classes[label]
            elif split == "validate":
                filePath += 'kinetics_val/'
                label = row['label']
                num_label = self.classes[label]
            elif split == "test":
                filePath += 'kinetics_test/'
                label = "test_NAN_label"
            else:
                print("Data type not recognized for file {}".format(name))
            filePath += '{}.mp4'.format(name)

            try:
                (sig, rate) = librosa.load(filePath, offset=start, duration=10)
                mfcc = librosa.feature.mfcc(y=sig, sr=rate, n_mfcc=13)
                chroma = librosa.feature.chroma_cqt(y=sig, sr=rate, n_chroma=12)
                return np.concatenate((mfcc, chroma)), num_label, 431 
                # return np.zeros((25, 431)), num_label, 431
            except:
                # print(filePath)
                return None, None, None
            # if idx % 2 == 0:
            #     return None, None, None
            # return np.zeros((25, 431)), num_label, 431


    def __init__(self):
        super(Net, self).__init__()
        self.num_features = 25
        self.input_size = 128
        self.seq_len = 431
        self.hidden_size = 80
        self.other_hidden_size = 80
        self.num_classes = 700
        self.num_epochs = 2
        self.batch_size = 20
        self.learning_rate = 0.00085
        self.attention_heads = 4
        self.n_layers = 4
        self.counter = 0

        self.trainPath = 'train'
        self.testPath = 'test'
        self.valPath = 'validate'
        self.classes = self.get_pickle("classes_dict")

        
        wandb.init()
        self.fc0 = nn.Linear(self.num_features, self.input_size)
        self.encode = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.attention_heads)
        self.transformer = nn.TransformerEncoder(self.encode, num_layers=self.n_layers)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size) 
        self.dropout = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(self.hidden_size, self.other_hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(self.other_hidden_size, self.num_classes)
        self.loss = nn.CrossEntropyLoss()

        self.train_test_split_pct = 0.8
        self.train_dataset, self.test_dataset, self.val_dataset = self.prepare_data()
        self.conf_target = ()
        self.conf_pred = ()
        self.hist_count = 0
        # self.train_hist = self.create_class_hists(self.train_dataset, "train.png")
        # self.test_hist = self.create_class_hists(self.test_dataset, "test.png")
     

    def get_pickle(self, classPath):
            with open('Desktop/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
                result = pickle.load(handle)
            return result

    def create_class_hists(self, data, name):
        hist_list = []
        for i in data.indices:
            hist_list.append(data.dataset[i][1])
        labels, values = zip(*Counter(hist_list).items())
        indexes = np.arange(len(labels))
        width = 1
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.savefig(name)
        
    
    def forward(self, x, mask):
        expand = self.fc0(x)
        transform = self.transformer(expand, src_key_padding_mask=mask)
        mean = self.dropout(self.fc1(transform.permute(1, 0, 2).mean(dim=1)))
        rel = self.relu(mean)
        out = self.fc3(rel)
        return out

    def prepare_data(self):
        train_dataset = self.AudioDataLoader(self.trainPath)
        test_dataset = self.AudioDataLoader(self.testPath)
        val_dataset = self.AudioDataLoader(self.valPath)
        return train_dataset, test_dataset, val_dataset
    
    def collate_fn(self, batch):
        # return torch.utils.data.dataloader.default_collate(batch)

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
        data = list(filter(lambda x: x is not None, batch[0]))
        labels = list(filter(lambda x: x is not None, batch[1]))


        # labels = batch[1].tolist()

        samples = torch.Tensor([(trp(t, self.seq_len))for t in data])
        mask = torch.Tensor([pad_masker(t) for t in samples])
        labels = torch.Tensor(labels).long()
        samples = samples.permute(1, 0, 2)

        return samples, labels, mask

    def train_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=True,
                                                        collate_fn=self.collate_fn,
                                                        num_workers=50)
        return self.train_loader


    def val_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=True,
                                                        collate_fn=self.collate_fn,
                                                        num_workers=50)        
        return self.test_loader

    def training_step(self, batch, batch_idx):
        # print(batch)
        data, target, mask = batch
        mask = mask < 0
        output = self.forward(data, mask)

        loss = self.loss(output, target)
        logs = {'loss': loss}
        # wandb_logger.agg_and_log_metrics(logs)
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

        # if self.counter >= self.num_epochs-1:
            # wandb.sklearn.plot_confusion_matrix(target.cpu().numpy(), pred.flatten().cpu().numpy(), self.classes)

        return {'val_loss': temp_loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        # print(outputs)
        self.counter += 1
        self.hist_count += 1
        # if self.counter == self.num_epochs - 1:
        #     wandb.sklearn.plot_confusion_matrix(np.hstack(self.conf_target), np.hstack(self.conf_pred), self.classes)

        avg_loss = torch.stack([m['val_loss'] for m in outputs]).mean()
        avg_acc = torch.stack([m['val_acc'] for m in outputs]).mean()
        logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=29, epochs=self.num_epochs)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.02)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,80, 85, 90, 95], gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer


if __name__ == '__main__':
    
    model = Net()
    # wandb.watch(model)
    # trainer = pl.Trainer(gpus=4, max_epochs=2, logger=wandb_logger)
    # trainer = pl.Trainer(default_root_dir='/home/sgurram/good-checkpoint/', gpus=4, max_epochs=100, logger=wandb_logger, precision=16)
    #https://github.com/NVIDIA/apex (precision=16)
    # trainer = pl.Trainer(default_root_dir='/home/sgurram/good-checkpoint/', gpus=4, max_epochs=100, logger=wandb_logger, distributed_backend='ddp2')
    trainer = pl.Trainer(default_root_dir='/home/sgurram/good-checkpoint/', gpus=[2,3], max_epochs=2, logger=wandb_logger, accumulate_grad_batches=2, distributed_backend='ddp')
    trainer.fit(model)
