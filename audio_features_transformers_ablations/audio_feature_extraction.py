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
import pydub
from pydub import AudioSegment


os.environ["IMAGEIO_FFMPEG_EXE"] = "/home/sgurram/anaconda3/bin/ffmpeg"
warnings.filterwarnings("ignore")


class AudioFeatureExtractor:
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

    def get_dataset_count(self, rootdir):
        count = 0
        for path in glob.glob(f'{rootdir}/**/*.mp4'):
            count += 1
        return count

    def get_file_size(self, path):
        if os.path.isfile(path): 
            return os.stat(path).st_size

    def get_file_size_hists(self, rootdir):

        sizes = []
        for path in glob.glob(f'{rootdir}/**/*.mp4'):
            sizes.append(self.get_file_size(path))

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
    
    def make_classes_list(self, rootdir):
        classes = []
        for subdir, dirs, files in os.walk(rootdir):
            classes.append(subdir)
        classes.remove(rootdir)
        return classes

    def create_class_subdirs(self, savedir):
        for subfolder_name in self.classes:
            os.makedirs(os.path.join(savedir, subfolder_name))

    def mp4_wav_pykaldi(self, row):
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
            audio = AudioSegment.from_file(filePath, format="mp4")[(start*1000):(end*1000)]
            saveDir = '/data3/kinetics_pykaldi/{}/{}_{}'.format(split, str(num_label), label)
            savePath = '{}/{}->-{}.wav'.format(saveDir, str(start), name)
            os.makedirs(saveDir, exist_ok=True)
            audio.export(savePath, format="wav")

        except:
            # print("Error processing file: {}".format(name))
            dummy = 2


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
            label = "test_NAN_label"
        else:
            print("Data type not recognized for file {}".format(name))
        filePath += '{}.mp4'.format(name)

        try:
            (sig, rate) = librosa.load(filePath, offset=start, duration=10)
            mfcc = librosa.feature.mfcc(y=sig, sr=rate, n_mfcc=13)
            chroma = librosa.feature.chroma_cqt(y=sig, sr=rate, n_chroma=12)
            saveDir = '/data2/kinetics_audio/{}/{}_{}'.format(split, str(num_label), label)
            savePath = '{}/{}->-{}.pickle'.format(saveDir, str(start), name)
            os.makedirs(saveDir, exist_ok=True)

            with open(savePath, 'wb') as handle:
                pickle.dump(np.concatenate((mfcc, chroma)), handle, protocol=4)
            # return (split, savePath, num_label, np.concatenate((mfcc, chroma)))

        except:
            # print("Error processing file: {}".format(name))
            dummy = 2

    def extract_mel_spec(self, row):
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
            saveDir = '/home/sgurram/kinetics/{}/{}/{}->-{}'.format(split, str(num_label), str(start), name)
            specPath = '{}/{}.png'.format(saveDir, "spectro")
            featPath = '{}/{}.pickle'.format(saveDir, "vector")
            os.makedirs(saveDir, exist_ok=True)

            (sig, rate) = librosa.load(filePath, offset=start, duration=10)
            fig = plt.figure(figsize=[0.5,0.5])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            S = librosa.feature.melspectrogram(y=sig, sr=rate, n_mels=512)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
            plt.savefig(specPath, dpi=400, bbox_inches='tight',pad_inches=0)
            plt.close('all')
            scaled_image_vector = np.array(Image.open(specPath + ".png").convert('RGB').resize((150, 150)))/255

            with open(featPath, 'wb') as handle:
                pickle.dump(scaled_image_vector, handle, protocol=4)

        except:
            # print("Error processing file: {}".format(name))
            dummy = 2

    def get_wav_files_parallel(self):
        labelDict = {}
        vectorDict = {}
        count = 0
        data_len = 600 
        files = []
   
        
        train_df = pd.read_csv("/home/sgurram/Desktop/{}.csv".format("train"))
        test_df =  pd.read_csv("/home/sgurram/Desktop/{}.csv".format("test"))
        val_df =  pd.read_csv("/home/sgurram/Desktop/{}.csv".format("validate"))
        frames = [train_df, test_df, val_df]
        master_df = pd.concat(frames)

        result = Parallel(n_jobs=12, backend='loky')(delayed(self.mp4_wav_pykaldi)(row) for index, row in tqdm(master_df[0:100000].iterrows()))
        result = Parallel(n_jobs=12, backend='loky')(delayed(self.mp4_wav_pykaldi)(row) for index, row in tqdm(master_df[100000:200000].iterrows()))
        result = Parallel(n_jobs=12, backend='loky')(delayed(self.mp4_wav_pykaldi)(row) for index, row in tqdm(master_df[200000:300000].iterrows()))
        result = Parallel(n_jobs=12, backend='loky')(delayed(self.mp4_wav_pykaldi)(row) for index, row in tqdm(master_df[300000:400000].iterrows()))
        result = Parallel(n_jobs=12, backend='loky')(delayed(self.mp4_wav_pykaldi)(row) for index, row in tqdm(master_df[400000:500000].iterrows()))
        result = Parallel(n_jobs=12, backend='loky')(delayed(self.mp4_wav_pykaldi)(row) for index, row in tqdm(master_df[500000:600000].iterrows()))
        result = Parallel(n_jobs=12, backend='loky')(delayed(self.mp4_wav_pykaldi)(row) for index, row in tqdm(master_df[600000:700000].iterrows()))
        return "Done"

    def get_audio_features_parallel(self):
        labelDict = {}
        vectorDict = {}
        count = 0
        data_len = 600 
        files = []
   
        
        train_df = pd.read_csv("/home/sgurram/Desktop/{}.csv".format("train"))
        test_df =  pd.read_csv("/home/sgurram/Desktop/{}.csv".format("test"))
        val_df =  pd.read_csv("/home/sgurram/Desktop/{}.csv".format("validate"))
        frames = [train_df, test_df, val_df]
        master_df = pd.concat(frames)

        # result = Parallel(n_jobs=12, backend='loky')(delayed(self.extract_mfcc_chroma)(row) for index, row in tqdm(master_df[0:100000].iterrows()))
        # result = Parallel(n_jobs=12, backend='loky')(delayed(self.extract_mfcc_chroma)(row) for index, row in tqdm(master_df[100000:200000].iterrows()))
        # result = Parallel(n_jobs=12, backend='loky')(delayed(self.extract_mfcc_chroma)(row) for index, row in tqdm(master_df[200000:300000].iterrows()))
        # result = Parallel(n_jobs=12, backend='loky')(delayed(self.extract_mfcc_chroma)(row) for index, row in tqdm(master_df[300000:400000].iterrows()))
        # result = Parallel(n_jobs=12, backend='loky')(delayed(self.extract_mfcc_chroma)(row) for index, row in tqdm(master_df[400000:500000].iterrows()))
        # result = Parallel(n_jobs=12, backend='loky')(delayed(self.extract_mfcc_chroma)(row) for index, row in tqdm(master_df[500000:600000].iterrows()))
        result = Parallel(n_jobs=12, backend='loky')(delayed(self.extract_mfcc_chroma)(row) for index, row in tqdm(master_df[600000:700000].iterrows()))

        # for path in glob.glob(f'{rootdir}/**/*.mp4'):
        #     files.append(path)
        # files = master_df.values.toList()
        # result_feats = Parallel(n_jobs=-1, backend='loky')(delayed(extract_mfcc_chroma)(files[i]) for i in tqdm(range(data_len)))
        # result_feats = Parallel(n_jobs=12, backend='loky')(delayed(self.extract_mfcc_chroma)(row) for index, row in tqdm(master_df[0:300000].iterrows()))
        # result_feats = Parallel(n_jobs=12, backend='loky')(delayed(self.extract_mfcc_chroma)(row) for index, row in tqdm(master_df[0:300000].iterrows()))
        # self.create_feature_dicts(result_feats)
        return "Done"

    def create_feature_dicts(self, features):
        for sample in features:
            if sample:
                if sample[2] in self.all_dict[sample[0]]:
                    self.all_dict[sample[0]][sample[2]][sample[1]] = sample[3]
                else:
                    self.all_dict[sample[0]][sample[2]] = {}
                    self.all_dict[sample[0]][sample[2]][sample[1]] = sample[3]

    def save_feature_dicts(self):
        for key, value in self.train_dict.items():
            with open('kinetics/train/{}.pickle'.format(key), 'wb') as handle:
                pickle.dump(value, handle, protocol=4)

        for key, value in self.test_dict.items():
            with open('kinetics/test/{}.pickle'.format(key), 'wb') as handle:
                pickle.dump(value, handle, protocol=4)

        for key, value in self.val_dict.items():
            with open('kinetics/val/{}.pickle'.format(key), 'wb') as handle:
                pickle.dump(value, handle, protocol=4)

    def deprecated_audio_features(self):
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

    def deprecated_mel_spec(rootdir, savedir):
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

    def deprecated_save_csv_to_files(self):
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

    def kl_divergence(self, p, q):
        return sum(p[i] * np.log(p[i]/(q[i]+0.1)) for i in range(len(p)))

    def angular_similarity(self, x,y):
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        cos = np.dot(x, y)/(nx * ny)
        if cos > 1:
            cos = 1
        elif cos < -1:
            cos = -1
        return 1 - np.arccos(cos)/np.pi
            


if __name__ == "__main__":
    afe = AudioFeatureExtractor("kinetics700")
    result = afe.get_wav_files_parallel()

    # afe.save_feature_dicts()
    # print(len(list(afe.train_dict.keys())))
    # with open('kinetics/train/27.pickle', 'rb') as handle:
    #     testing = pickle.load(handle)
    # (sig, rate) = librosa.load("/data2/kinetics/kinetics_val/-BkdwVmG-zw.mp4", offset=0, duration=10)
    # (sig, rate) = librosa.load("/data2/kinetics/kinetics_val/-586Cj6Npa8.mp4", offset=0, duration=10)
    # print(rate)

    # audio = AudioSegment.from_file("/data2/kinetics/kinetics_val/-BkdwVmG-zw.mp4", format="mp4")[0:(10*1000)]
    # audio.export("-BkdwVmG-zw.wav", format="wav")




# https://surfboard.readthedocs.io/en/latest/multiproc_feat.html#surfboard.feature_extraction_multiprocessing.extract_features_from_path
# https://stackoverflow.com/questions/55487391/unable-to-use-multithread-for-librosa-melspectrogram
# https://medium.com/@robertbracco1/how-to-do-fastai-even-faster-bebf9693c99a
