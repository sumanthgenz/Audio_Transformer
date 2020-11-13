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

import torchaudio
print(torch.__version__)
print(pl.__version__)

# import kaldi
# from kaldi.matrix import Matrix

# from kaldi.feat.mfcc import Mfcc, MfccOptions
# from kaldi.matrix import SubVector, SubMatrix
# from kaldi.util.options import ParseOptions
# from kaldi.util.table import SequentialWaveReader, RandomAccessWaveReader
# from kaldi.util.table import MatrixWriter
# from numpy import mean
# from sklearn.preprocessing import scale
# import kaldiio


wandb_logger = WandbLogger(name='audio_self_supervised_run',project='kinetics-ablation')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore")


# gc.collect()
# torch.cuda.empty_cache()



class Net(pl.LightningModule):

    class AudioDataLoader(Dataset):
    
        def __init__(self, csvType):
            self.csvType = csvType
            self.dir = "/ssd/kinetics_pykaldi/{}".format(csvType)
            self.num_classes = 700
            self.downsamp_factor = 4
            self.samp_freq = 16000
            self.seq_len = 500
            self.wav_paths = self.create_pykaldi_scp_file()

            # usage = """Extract MFCC features.
            # Usage:  example.py [opts...] <rspec> <wspec>
            # """
            # po = ParseOptions(usage)
            # po.register_float("min-duration", 0.0, "minimum segment duration")
            # mfcc_opts = MfccOptions()
            # mfcc_opts.frame_opts.samp_freq = 22050
            # mfcc_opts.register(po)
            # opts = po.parse_args()
           
            # self.rspec = "p,scp:kinetics_pykaldi_{}.scp".format(csvType)
            # self.mfcc = Mfcc(mfcc_opts)
            # self.wav_reader = RandomAccessWaveReader(self.rspec)
            
        def create_pykaldi_scp_file(self):
            wav_paths = []
            for path in glob.glob(f'{self.dir}/**/*.wav'):
                wav_paths.append(path)

            # count = 0
            # with open("kinetics_pykaldi_{}.scp".format(self.csvType), "w") as output:
            #     for one_path in wav_paths:
            #         output.write("{} {}\n".format(count, one_path))
            #         count += 1
            return wav_paths


        def get_pickle(self, classPath):
            with open('Desktop/kinetics/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
                result = pickle.load(handle)
            return result
        
        def __len__(self):
            return len(self.wav_paths)

        def getNumClasses(self):
            return self.num_classes

        def __getitem__(self, idx):
            try:
                filePath = self.wav_paths[idx]
                num_label = int((filePath.split('/')[4]).split('_')[0]) - 1

                # num_label = 1

                # wav = self.wav_reader[str(idx)]
                # signal = wav.data()[:,::self.downsamp_factor]
                # matrix = SubVector(mean(signal, axis=0))
                # feat = np.transpose(np.array(self.mfcc.compute_features(matrix, self.samp_freq, 1.0)))

                #Updated torchaudio MFCC
                wav, samp_freq = torchaudio.load(filePath)
                wav = (wav.mean(dim=0))[::self.downsamp_factor]
                feat = np.array((torchaudio.transforms.MFCC(sample_rate=self.samp_freq)(wav.unsqueeze(0))).mean(dim=0))
                # feat = np.transpose(np.array(torchaudio.compliance.kaldi.mfcc(wav, sample_frequency=self.samp_freq)))
                # print(feat.shape)
                return feat, num_label, self.seq_len

            except:
                return None, None, None

    def __init__(self):
        super(Net, self).__init__()
        self.num_features = 40
        self.input_size_half = 256
        self.input_size = 512
        self.seq_len = 500
        self.hidden_size_1 = 1024
        self.hidden_size_2 = 2048
        self.hidden_size_3 = 1024
        self.num_classes = 700
        self.num_epochs = 100
        self.batch_size = 8
        self.learning_rate = 0.00025
        self.attention_heads = 8
        self.n_layers = 16
        self.counter = 0

        self.trainPath = 'train'
        self.testPath = 'test'
        self.valPath = 'validate'
        self.classes = self.get_pickle("classes_dict")

        
        
        self.encode = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.attention_heads, dropout=0.15)
        self.transformer = nn.TransformerEncoder(self.encode, num_layers=self.n_layers)

        self.fc0 = nn.Linear(self.num_features, self.input_size_half)
        self.fc1 = nn.Linear(self.input_size_half, self.input_size)
        self.fc2 = nn.Linear(self.input_size, self.hidden_size_1) 
        self.fc3 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc4 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.fc5 = nn.Linear(self.hidden_size_3, self.num_classes)

        self.old_fc4 = nn.Linear(self.hidden_size_1, self.num_classes)


        self.norm1 = nn.LayerNorm(self.input_size_half)
        self.norm2 = nn.LayerNorm(self.input_size)
        self.norm3 = nn.LayerNorm(self.hidden_size_1)
        self.norm4 = nn.LayerNorm(self.hidden_size_2)
        self.norm5 = nn.LayerNorm(self.hidden_size_3)


        self.dropout = nn.Dropout(p=0.15)
        self.relu = nn.ReLU()

        self.loss = nn.CrossEntropyLoss()

        self.train_test_split_pct = 0.8
        self.train_dataset, self.test_dataset, self.val_dataset = self.prepare_data()
        self.conf_target = ()
        self.conf_pred = ()
        self.hist_count = 0
        # self.train_hist = self.create_class_hists(self.train_dataset, "train.png")
        # self.test_hist = self.create_class_hists(self.test_dataset, "test.png")
     

    def get_pickle(self, classPath):
            with open('Desktop/kinetics/kinetics_{}.pickle'.format(classPath), 'rb') as handle:
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
        # expand = self.fc0(x)
        # transform = self.transformer(expand, src_key_padding_mask=mask)
        # mean = self.dropout(self.fc2(transform.permute(1, 0, 2).mean(dim=1)))
        # rel = self.relu(mean)
        # out = self.old_fc4(rel)
        # return out

        expand = self.fc0(x)
        expand = self.norm1(self.dropout(expand))
        expand2 = self.fc1(expand)
        src = self.norm2(self.dropout(expand2))

        transform = self.transformer(src, src_key_padding_mask=mask)

        transform = self.relu(self.fc2(transform.permute(1, 0, 2).mean(dim=1)))
        transform = transform + self.dropout(transform)
        transform = self.norm3(transform)

        out = self.fc4(self.dropout(self.relu(self.fc3(transform))))
        out = transform + self.dropout(out)
        out = self.norm3(out)
        out = self.fc5(out)
        
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
        # print(batch[0][0].shape)
        data = list(filter(lambda x: x is not None, batch[0]))
        labels = list(filter(lambda x: x is not None, batch[1]))


        # labels = batch[1].tolist()

        samples = torch.Tensor([(trp(t, self.seq_len))for t in data])
        # print(samples.shape)
        mask = torch.Tensor([pad_masker(t) for t in samples])
        labels = torch.Tensor(labels).long()
        samples = samples.permute(1, 0, 2)

        return samples, labels, mask

    #Implementation taken from aai/utils/torch/metrics.py
    def _compute_mAP(self, logits, targets, threshold):  
        return torch.masked_select((logits > threshold) == (targets > threshold), (targets > threshold)).float().mean()

    #Implementation taken from aai/utils/torch/metrics.py
    def compute_mAP(self, logits, targets, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)): 
        return torch.mean(torch.stack([self._compute_mAP(logits, targets, t) for t in thresholds]))

    #Implementation taken from aai/utils/torch/metrics.py
    def compute_accuracy(self, logits, ground_truth, top_k=1):
        """Computes the precision@k for the specified values of k.
        https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py
        """
        batch_size = ground_truth.size(0)
        _, pred = logits.topk(top_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(ground_truth.reshape(1, -1).expand_as(pred))
        correct_k = correct[:top_k].reshape(-1).float().sum(0)
        return correct_k.mul_(100.0 / batch_size)

    def train_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=False,
                                                        collate_fn=self.collate_fn,
                                                        num_workers=12)
        return self.train_loader


    def val_dataloader(self):
        self.test_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=False,
                                                        collate_fn=self.collate_fn,
                                                        num_workers=12)        
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
        data, target, mask = batch
        mask = mask < 0
        output = self.forward(data, mask)
        temp_loss = self.loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        # print(pred)
        # print(target)

        # correct = pred.eq(target.view_as(pred)).sum().item()
        # acc = torch.tensor(correct/self.batch_size)

        # if correct > 0:
        #     print(acc)
        top_1_accuracy = self.compute_accuracy(output, target, top_k=1)
        top_5_accuracy = self.compute_accuracy(output, target, top_k=5)

        mAP = self.compute_mAP(torch.nn.functional.sigmoid(output),
                          torch.nn.functional.one_hot(target, num_classes=self.num_classes))

        logs = {
            'val_loss': temp_loss,
            'val_top_1': top_1_accuracy,
            'val_top_5': top_5_accuracy,
            'val_mAP' : mAP}

        # if self.counter >= self.num_epochs-1:
            # wandb.sklearn.plot_confusion_matrix(target.cpu().numpy(), pred.flatten().cpu().numpy(), self.classes)

        return logs
    
    def validation_epoch_end(self, outputs):
        # print(outputs)
        self.counter += 1
        self.hist_count += 1
        # if self.counter == self.num_epochs - 1:
        #     wandb.sklearn.plot_confusion_matrix(np.hstack(self.conf_target), np.hstack(self.conf_pred), self.classes)

        avg_loss = torch.stack([m['val_loss'] for m in outputs]).mean()
        avg_top1 = torch.stack([m['val_top_1'] for m in outputs]).mean()
        avg_top5 = torch.stack([m['val_top_5'] for m in outputs]).mean()
        avg_mAP = torch.stack([m['val_mAP'] for m in outputs]).mean()

        logs = {
        'val_loss': avg_loss,
        'val_top_1': avg_top1,
        'val_top_5': avg_top5,
        'val_mAP' : avg_mAP}

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
    wandb_logger.watch(model, log='gradients', log_freq=400)

    # wandb_logger.watch(model, log='gradients', log_freq=10)
    # trainer = pl.Trainer(gpus=4, max_epochs=2, logger=wandb_logger)
    # trainer = pl.Trainer(default_root_dir='/home/sgurram/good-checkpoint/', gpus=4, max_epochs=100, logger=wandb_logger, precision=16)
    #https://github.com/NVIDIA/apex (precision=16)
    # trainer = pl.Trainer(default_root_dir='/home/sgurram/good-checkpoint/', gpus=4, max_epochs=100, logger=wandb_logger, distributed_backend='ddp2')
    
    # trainer = pl.Trainer(
    #     default_root_dir='/home/sgurram/good-checkpoint/', 
    #     auto_lr_find=True, gpus=[2,3], 
    #     overfit_batches=10, 
    #     max_epochs=100, 
    #     logger=wandb_logger, 
    #     accumulate_grad_batches=1, 
    #     distributed_backend='ddp')

    # trainer = pl.Trainer(
    #     default_root_dir='/home/sgurram/Projects/audio_transformer_supervised/audio_features_transformers_ablations/good-checkpoint', 
    #     gpus=1, 
    #     overfit_batches=10, 
    #     max_epochs=5, 
    #     logger=wandb_logger, 
    #     accumulate_grad_batches=1)
   
    trainer = pl.Trainer(
        default_root_dir='/home/sgurram/Projects/audio_transformer_supervised/audio_features_transformers_ablations/good-checkpoint', 
        gpus=2, 
        max_epochs=50, 
        logger=wandb_logger, 
        accumulate_grad_batches=800,
        distributed_backend='ddp')    

    # trainer = pl.Trainer(
    #     default_root_dir='/home/sgurram/good-checkpoint/', 
    #     gpus=[0, 1, 2,3], 
    #     max_epochs=50, 
    #     logger=wandb_logger, 
    #     accumulate_grad_batches=200, 
    #     distributed_backend='ddp')

    # lr_finder = trainer.tuner.lr_find(model)
    # lr_finder.results
    # new_lr = lr_finder.suggestion()

    
    trainer.fit(model)
