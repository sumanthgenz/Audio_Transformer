import tensorflow as tf 
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from collections import Counter
import surfboard
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

os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")


class AudioFeatureExtractor:

    os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
    warnings.filterwarnings("ignore")
    def __init__(self, name):
        self.name = name
        self.classes = self.get_pickle("classes_dict")
        self.rootdir = '/data2/kinetics/'
        self.savdir = ''

        self.all_dict = {}
        self.train_dict = {}
        self.test_dict = {}
        self.val_dict = {}
        self.all_dict["train"] = self.train_dict
        self.all_dict["test"] = self.test_dict
        self.all_dict["validate"] = self.val_dict


   

    def get_pickle(self, path):
        with open('Desktop/kinetics_{}.pickle'.format(path), 'rb') as handle:
            result = pickle.load(handle)
        return result

    def save_files(self):
        files = []
        train_df = pd.read_csv("/home/sgurram/Desktop/{}.csv".format("train"))
        test_df =  pd.read_csv("/home/sgurram/Desktop/{}.csv".format("test"))
        val_df =  pd.read_csv("/home/sgurram/Desktop/{}.csv".format("validate"))
        for index, row in train_df.iterrows():
            files.append(row)
        for index, row in test_df.iterrows():
            files.append(row)
        for index, row in val_df.iterrows():
            files.append(row)
        with open('Desktop/kinetics_files1.npy', 'wb') as f:
            np.save(f, files[0:100000])
        with open('Desktop/kinetics_files2.npy', 'wb') as f:
            np.save(f, files[100000:200000])
        with open('Desktop/kinetics_files3.npy', 'wb') as f:
            np.save(f, files[200000:300000])
        with open('Desktop/kinetics_files4.npy', 'wb') as f:
            np.save(f, files[300000:400000])
        with open('Desktop/kinetics_files5.npy', 'wb') as f:
            np.save(f, files[400000:500000])
        with open('Desktop/kinetics_files6.npy', 'wb') as f:
            np.save(f, files[500000:len(files)])
        

    def enumerate_dataset(self, rootdir):
        count = 0
        for path in glob.glob(f'{rootdir}/**/*.mp4'):
            count += 1
        return count

    def file_size(self, path):
        if os.path.isfile(path): 
            return os.stat(path).st_size

    def file_size_hists(self, rootdir):

        sizes = []
        for path in glob.glob(f'{rootdir}/**/*.mp4'):
            sizes.append(self.file_size(path))

        fig, ax = plt.subplots(4, figsize=(40, 20))
        fig.tight_layout(pad=5.0)

        n_bins = 100
        lower = 10000
        mid = 20000000
        upper = 3500000000

        ax[0].set_title("Bins for 10 KB - 20 MB")
        ax[0].hist(sizes, n_bins, (lower, mid))

        ax[1].set_title("Bins for 20 MB - 3.5 GB")
        ax[1].hist(sizes, n_bins, (mid, upper))

        ax[2].set_title("Cumulative for 10 KB - 20 MB")
        ax[2].hist(sizes, n_bins, (lower, mid), cumulative=True)

        ax[3].set_title("Cumulative for 20 KB - 3.5 GB")
        ax[3].hist(sizes, n_bins, (mid, upper), cumulative=True)

        plt.savefig("kinetics_file_sizes.png")

    def extract_mfcc_chroma(self, row):
        # Surfboard Code
        # sound = Waveform(path=filePath, sample_rate=20000)
        # mfcc = sound.mfcc()
        # chroma = sound.chroma_cqt()
        # spec_entropy = sound.spectral_entropy()
        # spec_skew = sound.spectral_skewness()
        # return np.concatenate((mfcc, chroma, spec_entropy, spec_skew))

        # Librosa Code
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
        else:
            print("Data type not recognized for file {}".format(name))
        filePath += '{}.mp4'.format(name)
        savePath = filePath + "_start_" + str(start) 

        try:
            (sig, rate) = librosa.load(filePath, offset=start, duration=10)
            mfcc = librosa.feature.mfcc(y=sig, sr=rate, n_mfcc=13)
            chroma = librosa.feature.chroma_cqt(y=sig, sr=rate, n_chroma=12)
            # self.val_dict[savePath] = (num_label, np.concatenate((mfcc, chroma)))
            return (split, savePath, num_label, np.concatenate((mfcc, chroma)))
            # print((num_label, np.concatenate((mfcc, chroma)).shape))

        except:
            # print("File not found: {}".format(name))
            dummy = 2

    def kld(self, p, q):
        return sum(p[i] * np.log(p[i]/(q[i]+0.1)) for i in range(len(p)))

    def ang(self, x,y):
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        cos = np.dot(x, y)/(nx * ny)
        if cos > 1:
            cos = 1
        elif cos < -1:
            cos = -1
        return 1 - np.arccos(cos)/np.pi

    def store_classes(self, rootdir):
        classes = []
        for subdir, dirs, files in os.walk(rootdir):
            classes.append(subdir)
        classes.remove(rootdir)
        return classes

    def populate_spec_savedir(self, savedir):
        for subfolder_name in self.classes:
            os.makedirs(os.path.join(savedir, subfolder_name))
        
    def get_audio_features(self):
        warnings.filterwarnings("ignore")
        labelDict = {}
        vectorDict = {}
        count = 0
        data_len = 50
        with tqdm(total=data_len) as pbar:
            for path in glob.glob(f'{self.rootdir}/**/*.mp4'):
                if count < data_len:
                    pbar.update(1)
                    try:
                        count += 1
                        print(path)
                        # labelDict[path] = classes.index(subdir)
                        vectorDict[path] = self.extract_mfcc_chroma(path)
                    except:
                        print("bad file: " + path)
                
                else:
                    break
        return labelDict, vectorDict

    def get_audio_features_parallel(self):
        labelDict = {}
        vectorDict = {}
        count = 0
        data_len = 6000 
        files = []
        # for path in glob.glob(f'{rootdir}/**/*.mp4'):
        #     files.append(path)
        train_df = pd.read_csv("/home/sgurram/Desktop/{}.csv".format("train"))
        test_df =  pd.read_csv("/home/sgurram/Desktop/{}.csv".format("test"))
        val_df =  pd.read_csv("/home/sgurram/Desktop/{}.csv".format("validate"))
        frames = [train_df, test_df, val_df]
        master_df = pd.concat(frames)
        # result_feats = Parallel(n_jobs=-1, backend='loky')(delayed(extract_mfcc_chroma)(files[i]) for i in tqdm(range(data_len)))
        # result_feats = Parallel(n_jobs=-1, backend='loky', require='sharedmem')(delayed(self.extract_mfcc_chroma)(row) for index, row in tqdm(val_df[:50].iterrows()))
        result_feats = Parallel(n_jobs=-1, backend='loky')(delayed(self.extract_mfcc_chroma)(row) for index, row in tqdm(master_df[:data_len].iterrows()))
        # result_feats = [self.extract_mfcc_chroma(row) for index, row in val_df[:50].iterrows()]
        self.load_dicts(result_feats)
        # return labelDict, vectorDict

    def load_dicts(self, features):
        for sample in features:
            if sample:
                self.all_dict[sample[0]][sample[1]] = (sample[2], sample[3])

    def get_mel_spec(rootdir, savedir):
        spec2LabelDict = {}
        spec2VectorDict = {}
        count = 0
        data_len = 592792
        with tqdm(total=data_len) as pbar:
            for path in glob.glob(f'{rootdir}/**/*.mp4'):
                pbar.update(1)
                wav_path = path
                #might need subdir.split("/")[-1]
                spec_path = os.path.join(savedir, wav_path.split(rootdir)[1])

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
                    spec2LabelDict[spec_path] = self.classes.index(subdir.split("/")[-1])
                    spec2VectorDict[spec_path] = np.array(Image.open(spec_path + ".png").convert('RGB').resize((150, 150)))/255
            
                except:
                    print("bad file: " + wav_path)
                if count % 1000 == 0:
                    print("1000 done")
        return spec2LabelDict, spec2VectorDict
                
    def save_data(self):
        try:
            with open('{}.pickle'.format("kinetics_train_data"), 'wb') as handle:
                pickle.dump(self.train_dict, handle, protocol=4)

            with open('{}.pickle'.format("kinetics_test_data"), 'wb') as handle:
                pickle.dump(self.test_dict, handle, protocol=4)
            
            with open('{}.pickle'.format("kinetics_val_data"), 'wb') as handle:
                pickle.dump(self.val_dict, handle, protocol=4)

        except:
            print("Pickle 4 Failed")

            # with open('{}.pickle'.format("kinetics_val_data"), 'wb') as handle:
            #     pickle.dump(val_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    # afe = AudioFeatureExtractor("kinetics700")
    # result = afe.get_audio_features_parallel()
    # afe.save_data()
    # print(len(list(afe.train_dict.keys())))
    with open('kinetics_test_data.pickle', 'rb') as handle:
        testing = pickle.load(handle)

    print(list(testing.values())[0:5])
    print(list(testing.keys())[0:5])


# https://surfboard.readthedocs.io/en/latest/multiproc_feat.html#surfboard.feature_extraction_multiprocessing.extract_features_from_path
# https://stackoverflow.com/questions/55487391/unable-to-use-multithread-for-librosa-melspectrogram
# https://medium.com/@robertbracco1/how-to-do-fastai-even-faster-bebf9693c99a
