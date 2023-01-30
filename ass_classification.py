# Download URL
# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
import os
import shutil
from pathlib import Path
from os.path import exists
import numpy as np
import sys
import data_utils

from matplotlib import pyplot as plt
from matplotlib import use as mpl_use
from pyfiglet import Figlet

from utils import unpickle, connect, download_dataset


def get_one_hot_vector(y):
    """
    generating one hot vector for class labels
    """
    y = np.array(y)
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    # y_one_hot = y_one_hot.reshape(-1, 10, 1)
    return y_one_hot


class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt(
            np.sum(np.square(self.X_train), axis=1) + np.sum(np.square(X), axis=1)[:, np.newaxis] - 2 * np.dot(X,
                                                                                                               self.X_train.T))
        pass
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            sorted_dist = np.argsort(dists[i])
            closest_y = list(self.y_train[sorted_dist[0:k]])
            pass
            y_pred[i] = (np.argmax(np.bincount(closest_y)))
            pass
        return y_pred


f = Figlet(font='slant')
print(f.renderText('CS774 Assignment 1'))

# Check for internet
print(
    "Connected to Internet. Ready for duty." if connect() else "No Internet! Put Extracted Dataset in current directory")

dir = 'cifar-10-batches-py'
file_exists = exists('cifar-10-python.tar.gz')
dir_exists = exists(dir)
if file_exists:
    print('Previous File Exists. Removing File ...')
    os.unlink('cifar-10-python.tar.gz')
elif dir_exists:
    files_required = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5',
                      'test_batch']
    flag = True
    for i in files_required:
        path = Path(dir + f'/{i}')
        if not path.is_file():
            flag = False
            break
    if flag:
        print('Dataset already available')
    else:
        print('Dataset Incomplete. Re-download required')
        shutil.rmtree(dir)
else:
    print('Dataset Unavailable. Connecting to internet ...')
    download_dataset()

# Load the dataset

f = Figlet(font='digital')
print(f.renderText('Question 1: Loading the dataset'))

train_data, test_data, labels_mapping = unpickle(dir)

# np.set_printoptions(threshold=sys.maxsize)
# print(train_data['data'][0])

print("Total train data size:", train_data['data'].shape)
print("Total test data size:", test_data['data'].shape)
print("Labels available for CIFAR-10: ", labels_mapping[b'label_names'])

# Question 2: Image transformations

f = Figlet(font='digital')
print(f.renderText('Question 7: Classification (KNN)'))

# org_train_images = preprocessing(train_data)
# org_test_images = preprocessing(test_data)
#
# # Generating Augmented Images
# train_augmented_img, train_augmented_labels = get_augmented_images(org_train_images[:100], train_data['labels'])
# augmented_train_set = np.vstack([org_train_images, train_augmented_img])
# augmented_train_labels = train_data['labels'] + train_augmented_labels
# print("Original Data Shape: ", org_train_images.shape)
# print('Augmented Data Shape: ', train_augmented_img.shape)
# print('Size of New Training Data Set: ', len(augmented_train_set))
#
# print("Classification on Un-Augmented Datasets")
# train_labels = get_one_hot_vector(train_data['labels'])
# test_labels = get_one_hot_vector(test_data['labels'])

cifar10_dir = './cifar-10-batches-py'
X_train, y_train, X_test, y_test = data_utils.load_CIFAR10(cifar10_dir)

# Checking the size of the training and testing data
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# # Memory error prevention by subsampling data
#
# num_training = 10000
# mask = list(range(num_training))
# X_train = X_train[mask]
# y_train = y_train[mask]
#
# num_test = 1000
# mask = list(range(num_test))
# X_test = X_test[mask]
# y_test = y_test[mask]

# reshaping data and placing into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# classifier = KNearestNeighbor()
# classifier.train(X_train, y_train)
# dists = classifier.compute_distances(X_test)
# y_test_pred = classifier.predict_labels(dists, k=5)
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
#
# num_folds = 5
# k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
#
# X_train_folds = []
# y_train_folds = []
#
# X_train_folds = np.array_split(X_train, num_folds)
# y_train_folds = np.array_split(y_train, num_folds)
# k_to_accuracies = {}
#
# for k in k_choices:
#     k_to_accuracies[k] = []
#     for num_knn in range(0, num_folds):
#         X_test = X_train_folds[num_knn]
#         y_test = y_train_folds[num_knn]
#         X_train = X_train_folds
#         y_train = y_train_folds
#
#         temp = np.delete(X_train, num_knn, 0)
#         X_train = np.concatenate((temp), axis=0)
#         y_train = np.delete(y_train, num_knn, 0)
#         y_train = np.concatenate((y_train), axis=0)
#
#         classifier = KNearestNeighbor()
#         classifier.train(X_train, y_train)
#         dists = classifier.compute_distances(X_test)
#         y_test_pred = classifier.predict_labels(dists, k)
#
#         num_correct = np.sum(y_test_pred == y_test)
#         accuracy = float(num_correct) / num_test
#         #         print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
#         k_to_accuracies[k].append(accuracy)
#
# print("Printing our 5-fold accuracies for varying values of k:")
# print()
# for k in sorted(k_to_accuracies):
#     for accuracy in k_to_accuracies[k]:
#         print('k = %d, accuracy = %f' % (k, accuracy))

print(f.renderText('Question 7 (d): Classification (Decision Tree Classifier)'))

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd


def DTC(_X=None, _Xt=None):
    if _X is None:
        _X = X_train

    if _Xt is None:
        _Xt = X_test

    print("[DTC] Training")
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)

    print("[DTC] Training Accuracy")
    X_pred = dtc.predict(X_train)
    print(metrics.accuracy_score(y_train, X_pred))

    print("[DTC] Testing Accuracy")
    Xt_pred = dtc.predict(X_test)
    print(metrics.accuracy_score(y_test, Xt_pred))

DTC()
