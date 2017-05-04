import cv2
import os
import pandas as pd
import numpy as np
from utils.image_processor import random_transform
from scipy.misc import imresize
# # git+https://github.com/uqfoundation/pathos.git@master
# http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma


class DataSet(object):
    ######################################
    # Process data from CSV(s) into
    # Training, Validation data set
    # =============================
    ######################################
    CSV_HEADER = ['center', 'left', 'right', 'steer_angle', 'throttle', 'brake', 'speed']

    def __init__(self, log_path, img_dir_path, sequence=10):
        self.df = pd.read_csv(log_path, names=self.CSV_HEADER, index_col=False)
        self.img_path = img_dir_path
        self.sequence_len = sequence
        self.X_train = []
        self.y_train = []

    def build_train_data(self):
        """
        :return:
        """
        print('Loading training data...')
        images = []
        left_images = []
        right_images = []
        left_measurements = []
        right_measurements = []
        center_measurements = []

        for i in range(len(self.df)):
            for image in ['center', 'left', 'right']:
                img_file = self.df.loc[i][image].rsplit('/')[-1]  # Extract image file only
                img = cv2.imread(os.path.join(self.img_path, img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = imresize(img, 0.5)
                if image == 'center':
                    images.append(img)
                    center_measurements.append((self.df.loc[i]['steer_angle'], self.df.loc[i]['throttle'], self.df.loc[i]['brake']))
                if image == 'left':
                    left_images.append(img)
                    left_measurements.append((self.df.loc[i]['steer_angle'] + 0.30, self.df.loc[i]['throttle'], self.df.loc[i]['brake']))
                if image == 'right':
                    right_images.append(img)
                    right_measurements.append((self.df.loc[i]['steer_angle'] - 0.30, self.df.loc[i]['throttle'], self.df.loc[i]['brake']))

        self.y_train = np.concatenate((center_measurements, left_measurements, right_measurements))
        self.X_train = np.concatenate((images, left_images, right_images))
        #
        print("Data loaded.")
        self.X_train = np.asarray(self.X_train)
        print("Input shape: ", np.shape(self.X_train))
        print("Label shape: ", np.shape(self.y_train))
        return self.X_train, self.y_train

    def get_data(self):
        return self.X_train, self.y_train
