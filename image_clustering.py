# for loading/processing the images
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

from PIL import Image

CURRENT_DIR = os.getcwd()
IMAGE_FOLDER = "archive/flower_images/flower_images"


images_path = os.path.join(CURRENT_DIR, IMAGE_FOLDER)

# list of all filenames of images
flowers = []

with os.scandir(images_path) as files:
    for file in files:
        if file.name.endswith('.png'):
            flowers.append(file.name)

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)


def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224, 224))
    img = np.array(img)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


data = {}
p = os.path.join(CURRENT_DIR, 'features')

for flower in flowers:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(os.path.join(images_path, flower), model)
        data[flower] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p, 'wb') as file:
            pickle.dump(data, file)

filenames = np.array(list(data.keys()))

# list of just the features
feat = np.array(list(data.values()))

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1, 4096)

# get the unique image types
df = pd.read_csv(os.path.join(images_path, 'flower_labels.csv'))
label = df['label'].tolist()
unique_labels = list(set(label))

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# cluster feature vectors
kmeans = KMeans(n_clusters=len(unique_labels), random_state=22)
kmeans.fit(x)

# holds the cluster id and related images
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

# Dump Grouped filesnames to a file
with open(os.path.join(CURRENT_DIR, 'dump', 'config.groups'), 'wb') as config_dictionary_file:
    pickle.dump(groups, config_dictionary_file)
