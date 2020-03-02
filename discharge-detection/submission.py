import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import sklearn
import csv
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Activation, ZeroPadding2D, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras import backend as K
import os

K.set_image_dim_ordering('tf')


def extract_feature(file_name, trim=False):
    X, sample_rate = librosa.load(file_name)
    non_silent_index = 0
    if trim:
        rsme = librosa.feature.rmse(y=X)
        rsme = rsme / np.max(rsme)
        rsme = rsme.flatten()
        for j, e in enumerate(rsme):
            if e > 0.01:
                non_silent_index = j
                break
    melspec = librosa.feature.melspectrogram(y=X, sr=sample_rate)
    feat_mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128)
    feat_delta = librosa.feature.delta(feat_mfcc)
    feat_delta = feat_delta / np.max(feat_delta)
    feat_delta = sklearn.preprocessing.normalize(feat_delta)
    feat_delta = feat_delta.T
    melspec = melspec.T
    melspec = librosa.power_to_db(melspec / np.max(melspec))
    melspec = sklearn.preprocessing.normalize(melspec)
    features = np.asarray([melspec[non_silent_index:, :], feat_delta[non_silent_index:, :]])
    features = np.einsum('ijk->jki', features)
    return features


def get_labels(file_name):
    labels = {}
    with open(file_name, 'r') as label_file:
        label_reader = csv.reader(label_file)
        next(label_reader, None)
        for row in label_reader:
            file_name, label = row
            labels[file_name] = int(label)
    return labels


train_labels = get_labels('./training-data/train_labels.csv')
parent_dir = './'
tr_sub_dirs = ["training-data"]
ts_sub_dirs = ["test-data"]
all_features = []
all_labels = []
min_height = 10


def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav", trim=False, long_files=[]):
    all_features = []
    all_labels = []
    for _, sub_dir in enumerate(sub_dirs):
        print("processing in subdir {}".format(sub_dir))
        file_count = 0
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            file_count += 1
            if file_count % 1000 == 0:
                print("file counts {}...".format(file_count))
            (_, filename) = os.path.split(fn)
            if filename in long_files:
                continue
            label = train_labels.get(filename, 0)
            features = extract_feature(fn, trim)
            height = features.shape[0]
            for i in range(0, height - min_height + 1):
                ext_features = features[i:i + min_height, :]
                all_features.append(ext_features)
                all_labels.append(label)
    all_features = np.asarray(all_features)
    all_labels = np.asarray(all_labels)
    all_labels = all_labels.reshape(-1, 1)
    return all_features, all_labels


parse_audio_files(parent_dir, tr_sub_dirs)
print("done processing files")
all_features = np.asarray(all_features)
all_labels = np.asarray(all_labels)
all_labels = all_labels.reshape(-1, 1)
print(np.asarray(all_features).shape)
print(np.asarray(all_labels).shape)

X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels)
X_train = np.expand_dims(X_train, axis=3)

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Activation, ZeroPadding2D, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras import backend as K

K.set_image_dim_ordering('tf')

inputs = Input(shape=all_features.shape, dtype='float32')
act = 'relu'
pool_size = 2
n_batch_size = 500
n_epochs = 2
model = Sequential()
model.add(Conv2D(64, (5, 5), padding='same', activation=act, input_shape=(10, 128, 1)))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Conv2D(64, (5, 5), padding='same', activation=act))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
model.add(Flatten())
# model.add(Dense(1000, activation=act))
model.add(Dense(100, activation=act))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=n_batch_size, epochs=n_epochs)

## check on test data
final_pred = []
test_predictions = {}

path = "test-data/"
test_list = os.listdir(path)
test_list.sort()
print(test_list)
for j in test_list:
    test_file = os.path.join(path, j)
    features = extract_feature(test_file)
    height = features.shape[0]
    label = 0
    feat = []
    for i in range(0, height - min_height + 1):
        ext_features = features[i:i + min_height, :]
        feat.append(ext_features)

    #         all_features.append(ext_features)
    feat = np.expand_dims(feat, axis=3)
    predict = model.predict(feat)
    a = np.copy(predict)
    a[a < 1e-5] = 0
    a[a >= 1e-5] = 1
    a = a.astype("uint8")
    a = np.sum(a)
    prediction = '0'
    if a >= 1:
        prediction = '1'
    final_pred.append(prediction)
    print("processing - {} - {}".format(j, prediction))
    test_predictions[j] = prediction

# test_predictions is the final prediction

all_test_features = {}
for fn in test_list:
    test_file = os.path.join(path, fn)
    features = extract_feature(test_file)
    height = features.shape[0]
    feat = []
    for i in range(0, height - min_height + 1):
        ext_features = features[i:i + min_height, :]
        feat.append(ext_features)
    feat = np.asarray(feat)
    all_test_features[fn] = feat


def write_inference_result(model, filename, threshold=0.1):
    all_predictions = {}
    with open(filename, 'w') as file:
        file.write('name,gunshot\n')
        for fn, features in all_test_features.items():
            predict = model.predict(features)
            prediction = '0'
            a = np.copy(predict)
            a[a < threshold] = 0
            a[a >= threshold] = 1
            a = a.astype("uint8")
            a = np.sum(a)
            if a >= 1:
                prediction = '1'
            all_predictions[fn] = prediction
            file.write('{},{}\n'.format(fn, prediction))
    print('writing inference result to file {} done'.format(filename))
    return all_predictions


write_inference_result(model, 'final_result.csv', 0.1)
