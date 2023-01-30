# Download URL
# https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
import os
import shutil
from pathlib import Path
from os.path import exists
import numpy as np
import sys

import torch
from matplotlib import pyplot as plt
from matplotlib import use as mpl_use
from pyfiglet import Figlet

from feature_extractor import BBResNet18
from mlp import get_one_hot_vector, Model
from utils import unpickle, connect, download_dataset, preprocessing, img_enhancement, random_rotation, \
    contrast_and_flip, img_posterization, get_augmented_images, get_feat_vec

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
print(f.renderText('Question 2 (a): Image Enhancement'))

org_train_images = preprocessing(train_data)
org_test_images = preprocessing(test_data)

# Test Image for applying image enhancement
first_image = org_train_images[0]

# Matplotlib Backend Specify to ignore the bug (MacOSX)
# mpl_use('MacOSX')
plt.figure(figsize=(10, 10))
# plt.imshow(first_image)
# plt.show()

# Question 2 (a)
enhanced_image = img_enhancement(first_image)
# plt.imshow(enhanced_image)
# plt.show()

# Question 2 (b)
print(f.renderText('Question 2 (b): Posterization of Image'))
posterized_image = img_posterization(first_image)
# plt.imshow(posterized_image)
# plt.show()

# Question 2 (c)
print(f.renderText('Question 2 (c): Random Rotate'))
rotated_image, rotated_degree = random_rotation(first_image)
# plt.imshow(rotated_image)
# plt.show()

print(f.renderText('Question 2 (d): Contrast and Horizontal Flipping'))
contrast_image, alpha = contrast_and_flip(first_image)
# plt.imshow(contrast_image)
# plt.show()

print(f.renderText('Question 3: Augmented Images'))

# Generating Augmented Images
cifar_augmented_dir = './cifar-10-batches-augmented-py'
print('Checking for preprocessed augmented data')
if exists(cifar_augmented_dir + '/augmented_batch.npy') and exists(cifar_augmented_dir + '/augmented_batch_labels.npy'):
    print('Preprocessed augmented data found. Loading Data')
    augmented_train_set = np.load(cifar_augmented_dir + '/augmented_batch.npy')
    augmented_train_labels = np.load(cifar_augmented_dir + '/augmented_batch_labels.npy')
else:
    print('Preprocessed augmented data NOT found. Regenerating Data')
    train_augmented_img, train_augmented_labels = get_augmented_images(org_train_images, train_data['labels'])
    augmented_train_set = np.vstack([org_train_images, train_augmented_img])
    augmented_train_labels = train_data['labels'] + train_augmented_labels
    np.save(cifar_augmented_dir + '/augmented_batch.npy', augmented_train_set)
    np.save(cifar_augmented_dir + '/augmented_batch_labels.npy', augmented_train_labels)
    print("Original Data Shape: ", org_train_images.shape)
    print('Augmented Data Shape: ', train_augmented_img.shape)

print('Size of New Training Data Set: ', len(augmented_train_set))
print(f.renderText('Question 4: Feature Vector'))

obj = BBResNet18()


if exists('./feature_vectors/original_train_feature_vector.npy') and exists('./feature_vectors/original_test_feature_vector.npy'):
    print("Loading Original Data Training Feature Vector")

    original_train_feat_vec = np.load('./feature_vectors/original_train_feature_vector.npy')
    original_test_feat_vec = np.load('./feature_vectors/original_test_feature_vector.npy')
else:
    print("Generating Original Data Training Feature Vector")

    original_train_feat_vec = get_feat_vec(org_train_images, obj)
    original_test_feat_vec = get_feat_vec(org_test_images, obj)

    np.save('./feature_vectors/original_train_feature_vector.npy', original_train_feat_vec)
    np.save('./feature_vectors/original_test_feature_vector.npy', original_test_feat_vec)

print(f.renderText('Question 5 & 6(a): MLP Implementation'))

labels = np.arange(10)
print("Training on Un-Augmented Datasets")
train_labels = get_one_hot_vector(train_data['labels'])
test_labels = get_one_hot_vector(test_data['labels'])
unaugmented_model = Model(original_train_feat_vec, train_labels, original_test_feat_vec, test_labels, './model_weights',
                          './output', isModelWeightsAvailable=1, epochs=500, batch_size=128, learning_rate=0.01,
                          augmented=False)
torch.save(unaugmented_model, './models/unaugmented_model')
print(f.renderText('Question 6 (b): Back Propagation'))

print("Checking Augmented Data Training Feature Vector")

if exists('./feature_vectors/augmented_train_feature_vector.npy'):
    print("Loading Augmented Data Training Feature Vector")

    augmented_train_feat_vec = np.load('./feature_vectors/original_train_feature_vector.npy')
else:
    print("Generating Augmented Data Training Feature Vector")
    augmented_train_feat_vec = get_feat_vec(augmented_train_set, obj)
    np.save('./feature_vectors/augmented_train_feature_vector.npy', augmented_train_feat_vec)

print("Training on Augmented Datasets")

aug_train_labels = get_one_hot_vector(augmented_train_labels)
augmented_model = Model(augmented_train_feat_vec, aug_train_labels, original_test_feat_vec, test_labels,
                        './model_weights',
                        './output', isModelWeightsAvailable=0, epochs=500, batch_size=128, learning_rate=0.01,
                        augmented=True)
torch.save(augmented_model, './models/augmented_model')
