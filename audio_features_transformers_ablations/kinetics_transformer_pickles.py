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


wandb_logger = WandbLogger(name='Remote_Audio_Transformer',project='kinetics-audio-feature-extraction')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# gc.collect()
# torch.cuda.empty_cache()

class Net(pl.LightningModule):

    class AudioDataLoader(Dataset):
    
        def __init__(self, csvType):
            self.dir = "/data2/kinetics_audio/{}".format(csvType)
            self.files = []
            self.seq_len = 431
            self.num_classes = 700
            try:
                for path in glob.glob(f'{self.dir}/**/*.pickle'):
                  self.files.append(path)
                print("Number of {} Samples: {}".format(csvType, len(self.files)))
            except:
                print("Error processing {}".format(csvType))

        def get_pickle(self, path):
            with open(path, 'rb') as handle:
                result = pickle.load(handle)
            return result
        
        def __len__(self):
            return len(self.files)

        def getNumClasses(self):
            return self.num_classes

        def __getitem__(self, idx):
            try:
                filePath = self.files[idx]
                num_label = int((filePath.split('/')[4]).split('_')[0]) - 1
                vectors = self.get_pickle(filePath)
                return vectors, num_label, self.seq_len
            except:
                print("bad processing for audioloader")


    def __init__(self):
        super(Net, self).__init__()
        self.num_features = 25
        self.input_size = 256
        self.seq_len = 431
        self.hidden_size = 512
        self.other_hidden_size = 1024
        self.num_classes = 700
        self.num_epochs = 500
        self.batch_size = 16
        self.learning_rate = 0.000025
        self.attention_heads = 4
        self.n_layers = 4
        self.counter = 0

        self.trainPath = 'train'
        self.testPath = 'test'
        self.valPath = 'validate'
        # self.classes = self.get_pickle("classes_dict")

        
        wandb.init()
        self.fc0 = nn.Linear(self.num_features, self.input_size)
        self.encode = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.attention_heads)
        self.transformer = nn.TransformerEncoder(self.encode, num_layers=self.n_layers)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size) 
        self.dropout = nn.Dropout(p=0.15)
        self.fc2 = nn.Linear(self.hidden_size, self.other_hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(self.other_hidden_size, self.other_hidden_size)
        self.fc4 = nn.Linear(self.other_hidden_size, self.num_classes)
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
        relu_1 = self.relu(mean)
        out_1 = self.dropout(self.fc2(relu_1))
        relu_2 = self.relu(out_1)
        out_2 = self.dropout(self.fc3(relu_2))
        relu_3 = self.relu(out_2)
        out_3 = self.dropout(self.fc3(relu_3))
        relu_4 = self.relu(out_3)
        out_4 = self.dropout(self.fc3(relu_3))
        relu_5 = self.relu(out_3)
        out_5 = self.fc4(relu_5)
        return out_5

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
                                                        num_workers=16)
        return self.train_loader


    def val_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=True,
                                                        collate_fn=self.collate_fn,
                                                        num_workers=16)        
        return self.test_loader

    def training_step(self, batch, batch_idx):
        data, target, mask = batch
        mask = mask < 0
        output = self.forward(data, mask)

        loss = self.loss(output, target)
        logs = {'loss': loss}
        # wandb_logger.agg_and_log_metrics(logs)
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        # wandb.init()
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
    
    trainer = pl.Trainer(default_root_dir='/home/sgurram/good-checkpoint/', overfit_batches=50, gpus=[2,3], max_epochs=200, logger=wandb_logger, accumulate_grad_batches=1, distributed_backend='ddp')
    trainer.fit(model)

    # dir = "/data2/kinetics_audio/{}".format("validate")
    # files = []
    # for path in glob.glob(f'{dir}/**/*.pickle'):
    #     files.append(path)
    # filePath = files[9000]
    # num_label = (filePath.split('/')[4]).split('_')[0]
    # with open(filePath, 'rb') as handle:
    #     result = pickle.load(handle)
    # print(filePath)
    # print(num_label)
    # print(result.shape)