# Function for downloading the dataset
import math
import urllib

import cv2
import numpy as np
import os
import sys
import random

import requests
from tqdm import tqdm


def download_dataset():
    print('Downloading CIFAR Dataset')
    import tarfile
    print('Download Complete. Extracting ...')
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    response = requests.get(url, stream=True)
    file = tarfile.open(fileobj=response.raw, mode="r|gz")
    file.extractall(path=".")
    print('Extraction Completed')


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)



# Function for checking the internet connectivity

def connect(host='http://google.com'):
    try:
        urllib.request.urlopen(host)  # Python 3.x
        return True
    except:
        return False

def url_exists(path):
    r = requests.head(path)
    return r.status_code == requests.codes.ok


# Unpack the dataset using pickle library
def unpickle(folder):
    import pickle
    labels_mapping = {}
    # np.set_printoptions(threshold=sys.maxsize)
    train_data = {'data': np.array([]), 'labels': []}
    test_data = {'data': np.array([]), 'labels': []}
    print(os.listdir(folder))
    for file in os.listdir(folder):
        if file == "data_batch_1" or file == "data_batch_2" or file == "data_batch_3" or file == "data_batch_4" or file == "data_batch_5":
            print('Currently Processing File : ', file)
            with open(os.path.join(folder, file), 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            if train_data['data'].shape[0] == 0:
                train_data['data'] = dict[b'data']
            else:
                train_data['data'] = np.vstack([train_data['data'], dict[b'data']])
            train_data['labels'] = train_data['labels'] + dict[b'labels']
        elif file == 'batches.meta':
            print('Currently Processing File : ', file)
            with open(os.path.join(folder, file), 'rb') as fo:
                labels_mapping = pickle.load(fo, encoding='bytes')
        elif file == 'test_batch':
            print('Currently Processing File : ', file)
            with open(os.path.join(folder, file), 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            if test_data['data'].shape[0] == 0:
                test_data['data'] = dict[b'data']
            test_data['labels'] = test_data['labels'] + dict[b'labels']
    return train_data, test_data, labels_mapping
    # with open(file, 'rb') as fo:
    #     dict = pickle.load(fo, encoding='bytes')
    # return dict


def preprocessing(data):
    images = []
    count = 1
    for img in data['data']:
        img_new = img.reshape((3, 32, 32))
        # print(img_new[0][0][0])
        image = np.transpose(img_new, [1, 2, 0])
        images.append(image)
    return np.array(images)


def img_enhancement(img):
    # print(img)
    arr_pxl = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            avg = np.average(img[i, j])
            arr_pxl.append(avg)
    imax = np.max(arr_pxl)
    imin = np.min(arr_pxl)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pxl = []
            avg = np.average(img[i, j])
            ip = 255 * (avg - imin) / (imax - imin)
            for c in range(img.shape[2]):
                if img[i, j, c] == 0:
                    n_pxl = 0
                else:
                    n_pxl = int((img[i, j, c] * ip / avg))
                pxl.append(n_pxl)
            img[i, j, :] = np.array(pxl)
            # print(img.shape)
    return img, img.shape


def img_posterization(img):
    random_x = random.randint(30, 50)
    random_y = random.randint(100, 150)
    imin = min(random_x, random_y)
    imax = max(random_x, random_y)
    range_pxl = abs(imax - imin)
    divider = 255 / range_pxl

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pxl = []
            for c in range(img.shape[2]):
                cont_pxl = (img[i, j, c] / divider)
                cont_pxl = (cont_pxl + imin)
                pxl.append(cont_pxl)
            img[i, j, :] = np.array(pxl).clip(0, 255)
    return img, img.shape


# This function rotates the image around its center by random degree between [-180, 180].
def random_rotation(image):
    # Choose Random degree
    degree = random.randint(-180, 180)
    # print("Random degree chosen: ", degree)
    # First we will convert the degrees into radians
    rads = math.radians(degree)
    cosine = math.cos(rads)
    sine = math.sin(rads)

    # Find the height and width of the rotated image using cosine and sine transformations
    height_rot_img = round(abs(image.shape[0] * cosine)) + round(abs(image.shape[1] * sine))
    width_rot_img = round(abs(image.shape[1] * cosine)) + round(abs(image.shape[0] * sine))

    # Initialising the rotated image by zeros
    rot_img = np.uint8(np.zeros((height_rot_img, width_rot_img, image.shape[2])))

    # Finding the center point of the original image
    orgx, orgy = (image.shape[1] // 2, image.shape[0] // 2)

    # Finding the center point of rotated image.
    rotx, roty = (width_rot_img // 2, height_rot_img // 2)

    for i in range(rot_img.shape[0]):
        for j in range(rot_img.shape[1]):
            # Find the all new coordinates for orginal image wrt the new center point
            x = (i - rotx) * cosine + (j - roty) * sine
            y = -(i - rotx) * sine + (j - roty) * cosine

            x = round(x) + orgy
            y = round(y) + orgx

            # Restricting the index in between original height and width of image.
            if x >= 0 and y >= 0 and x < image.shape[0] and y < image.shape[1]:
                rot_img[i, j, :] = image[x, y, :]
    return rot_img, degree


def contrast_and_flip(image):
    img_height = image.shape[0]
    img_width = image.shape[1]
    # Adding two pixels padding across the image
    img = np.uint8(np.zeros((img_height, img_width, image.shape[2])))

    alpha = random.uniform(0.5, 2.0)
    flip_prob = random.randint(0, 1)
    # print("Alpha value: ", alpha)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pxl = []
            for c in range(image.shape[2]):
                cont_pxl = int(alpha * (image[i, j, c] - 128) + 128)
                if cont_pxl > 255:
                    cont_pxl = 255
                elif cont_pxl < 0:
                    cont_pxl = 0
                pxl.append(cont_pxl)
            img[i, j, :] = np.array(pxl)
    if (flip_prob):
        # print("Including Horizontal Flipping")
        img = img[:, ::-1, :]  # Horizontal Flipping
    return img, round(alpha, 3)


# Generating Augmented Images
def get_augmented_images(data, labels):
    augmented_img = []
    augmented_labels = []
    preprocess_func = {0: random_rotation, 1: img_enhancement, 2: img_posterization, 3: contrast_and_flip}
    i = 0
    # print(data)
    for img in data:
        rndm_idx = random.randint(0, 3)
        if i % 1000 == 0:
            print("\nProcessing Image Number: ", i, end=' ')
        # Resizing to restore rotated image's dimensions to 32 x 32
        if preprocess_func[rndm_idx] != random_rotation:
            n_img, _ = preprocess_func[rndm_idx](img)
        else:
            # print(img)
            n_img, _ = preprocess_func[rndm_idx](img)
            n_img = cv2.resize(n_img, (32, 32))
        augmented_img.append(n_img)
        augmented_labels.append(labels[i])
        i += 1
    return np.array(augmented_img), augmented_labels


def get_feat_vec(images, obj):
    feat_vec = []
    count = 1
    for img in images:
        # print(count)
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 1, 0))
        # Performing Normalization before sending into ResNet model
        img = img / 255
        img = np.array(img, dtype=np.float32)
        feat_vec.append(obj.feature_extraction(np.array([img]))[0])
        count += 1
    return np.array(feat_vec)
