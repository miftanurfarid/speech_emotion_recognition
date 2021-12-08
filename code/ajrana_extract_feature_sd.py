# extract_feature_sd.py: to extract spectral features for speaker dependent

import glob
import os
import librosa
import numpy as np

data_path = '../data/song' # choose song or speech
files = glob.glob(os.path.join(data_path + '/*/', '*.wav'))
files.sort()

# function to extract feature
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name, sr=None)
    stft = np.abs(librosa.stft(X))
    mfcc = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    return (mfcc)


# create empty list to store features and labels
feat = []
lab = []

# iterate over all files
for file in files:
    print("Extracting features from ", file)
    feat_i = np.hstack(extract_feature(file))
    lab_i = os.path.basename(file).split('-')[2]
    feat.append(feat_i)
    lab.append(int(lab_i)-1)  # make labels start from 0

np.save(data_path + '/x_ajrana.npy', feat)
np.save(data_path + '/y_ajrana.npy', lab)
